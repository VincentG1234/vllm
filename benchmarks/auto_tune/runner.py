# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server and benchmark runner for vLLM auto-tuning."""

import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import regex as re
import requests
from config import TuneConfig
from workload import build_dataset_args


class ServerManager:
    """Manages vLLM server lifecycle."""

    def __init__(
        self,
        config: TuneConfig,
        log_dir: Path,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        profile_dir: Path | None = None,
    ):
        self.config = config
        self.log_dir = log_dir
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.profile_dir = profile_dir
        self.process: subprocess.Popen | None = None
        self.server_log_path = log_dir / "server.log"

    def _build_server_args(self) -> list[str]:
        """Build command-line arguments for vllm serve."""
        args = [
            "vllm",
            "serve",
            self.config.model.name,
            "--disable-log-requests",
            "--port",
            str(self.config.server.port),
            "--host",
            self.config.server.get_host(),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--tensor-parallel-size",
            str(self.config.model.tensor_parallel_size),
            "--max-model-len",
            str(self.config.model.max_model_len),
        ]

        if self.config.server.enable_prefix_caching:
            args.append("--enable-prefix-caching")

        if self.config.server.load_format:
            args.extend(["--load-format", self.config.server.load_format])

        if self.config.model.download_dir:
            args.extend(["--download-dir", self.config.model.download_dir])

        if self.profile_dir:
            profile_config = {
                "profiler": "torch",
                "torch_profiler_dir": str(self.profile_dir),
            }
            args.extend(["--profiler-config", json.dumps(profile_config)])

        return args

    def start(self, timeout_seconds: int = 600) -> bool:
        """
        Start the vLLM server and wait for it to be ready.

        Args:
            timeout_seconds: Maximum time to wait for server to start

        Returns:
            True if server started successfully, False otherwise
        """
        # Kill any existing vllm serve processes
        self._kill_existing_servers()

        # Ensure log directory exists before writing server.log (handles str or Path)
        Path(self.server_log_path).parent.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = self._build_server_args()
        env = os.environ.copy()
        env["VLLM_SERVER_DEV_MODE"] = "1"

        # Start server
        with open(self.server_log_path, "w") as log_file:
            self.process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,  # Create new process group
            )

        # Wait for server to be ready
        health_url = (
            f"http://{self.config.server.get_host()}:{self.config.server.port}/health"
        )
        max_attempts = timeout_seconds // 10
        for attempt in range(max_attempts):
            # Check if process is still alive
            if self.process.poll() is not None:
                print(f"Server process died. Check logs at {self.server_log_path}")
                return False

            # Check health endpoint
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(
                        f"Server started successfully on port {self.config.server.port}"
                    )
                    return True
            except requests.RequestException:
                pass

            time.sleep(10)

        print(
            f"Server did not start within {timeout_seconds} seconds. "
            f"Check logs at {self.server_log_path}"
        )
        return False

    def stop(self):
        """Stop the vLLM server."""
        if self.process is None:
            return

        # Kill the process group (including children)
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            # Wait a bit for graceful shutdown
            self.process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            # Process already dead or didn't terminate gracefully
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

        self.process = None
        time.sleep(2)  # Give port time to free up

    def _kill_existing_servers(self):
        """Kill any existing vllm serve processes."""
        # Use pkill to find and kill vllm serve processes
        subprocess.run(
            ["pkill", "-f", "vllm serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)

    @contextmanager
    def managed(self, timeout_seconds: int = 600):
        """Context manager for server lifecycle."""
        started = self.start(timeout_seconds)
        if not started:
            raise RuntimeError("Failed to start server")
        try:
            yield self
        finally:
            self.stop()


class BenchmarkRunner:
    """Runs vLLM benchmark and parses results."""

    def __init__(self, config: TuneConfig, log_dir: Path):
        self.config = config
        self.log_dir = log_dir
        self.bench_log_path = log_dir / "bench.log"

    def _build_bench_args(
        self,
        request_rate: float | None = None,
        prefix_len: int = 0,
        use_oversample: bool = True,
        profile: bool = False,
    ) -> list[str]:
        """Build command-line arguments for vllm bench serve."""
        args = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--model",
            self.config.model.name,
        ]

        # Add dataset arguments
        args.extend(build_dataset_args(self.config.workload))

        # Add common benchmark arguments
        args.extend(
            [
                "--ignore-eos" if self.config.run.ignore_eos else None,
                "--disable-tqdm" if self.config.run.disable_tqdm else None,
                "--percentile-metrics",
                ",".join(self.config.run.percentile_metrics),
                "--goodput",
                f"e2el:{self.config.constraints.max_p99_e2el_ms}",
                "--host",
                self.config.server.get_host(),
                "--port",
                str(self.config.server.port),
            ]
        )

        # Remove None values
        args = [a for a in args if a is not None]

        # Add request rate
        if request_rate is not None:
            if request_rate == float("inf"):
                args.extend(["--request-rate", "inf"])
            else:
                args.extend(["--request-rate", str(request_rate)])

        # Handle oversample flag
        if not use_oversample:
            args.append("--no-oversample")

        # Add prefix len (for random datasets)
        if self.config.workload.dataset_type == "random":
            args.extend(["--random-prefix-len", str(prefix_len)])

        # Add save-result to get JSON output
        args.extend(["--save-result", "--save-detailed"])

        # Save results to log_dir for easy access
        args.extend(["--result-dir", str(self.log_dir)])

        # Enable benchmark-level profiling (used for final best-config run)
        if profile:
            args.append("--profile")

        return args

    def run(
        self,
        request_rate: float | None = None,
        prefix_len: int = 0,
        use_oversample: bool = True,
        profile: bool = False,
    ) -> tuple[dict, bool]:
        """
        Run the benchmark.

        Args:
            request_rate: Request rate to use (None means use config default)
            prefix_len: Prefix length for cache simulation
            use_oversample: Whether to oversample the dataset
            profile: If True, pass --profile to vllm bench serve (final profiling run)

        Returns:
            Tuple of (results_dict, success)
        """
        if request_rate is None:
            if self.config.run.request_rate_mode == "fixed":
                request_rate = self.config.run.request_rate
            else:
                request_rate = float("inf")

        args = self._build_bench_args(
            request_rate=request_rate,
            prefix_len=prefix_len,
            use_oversample=use_oversample,
            profile=profile,
        )

        # Run benchmark
        with open(self.bench_log_path, "w") as log_file:
            result = subprocess.run(
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if result.returncode != 0:
            return {}, False

        # Try to parse JSON result file
        # vllm bench serve saves results to a JSON file when --save-result is used
        # Filename format: {label}-{request_rate}qps-{base_model_id}-{timestamp}.json
        # Find the most recent JSON file in the current directory or result_dir
        result_json = self._parse_benchmark_output()

        return result_json, True

    def _parse_benchmark_output(self) -> dict:
        """
        Parse benchmark output to extract metrics.

        First tries to find JSON output file, then falls back to parsing text logs.
        """
        # Try to find JSON result file in log directory (where we told bench to save it)
        json_files = list(self.log_dir.glob("*.json"))

        # Also check current directory as fallback
        json_files.extend(Path(".").glob("*.json"))

        if json_files:
            # Get the most recent JSON file
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            # Only use if it was created recently (within last 5 minutes)
            if time.time() - latest_json.stat().st_mtime < 300:
                with open(latest_json) as f:
                    return json.load(f)

        # Fallback: parse text output
        return self._parse_text_output()

    def _parse_text_output(self) -> dict:
        """Parse text output from bench log file."""
        results = {}
        if not self.bench_log_path.exists():
            return results

        with open(self.bench_log_path) as f:
            content = f.read()

        # Parse throughput
        throughput_match = re.search(
            r"Request throughput \(req/s\):\s*([0-9.]+)", content
        )
        if throughput_match:
            results["request_throughput"] = float(throughput_match.group(1))

        # Parse P99 E2EL
        e2el_match = re.search(r"P99 E2EL \(ms\):\s*([0-9.]+)", content)
        if e2el_match:
            results["p99_e2el_ms"] = float(e2el_match.group(1))

        # Parse goodput
        goodput_match = re.search(r"Request goodput \(req/s\):\s*([0-9.]+)", content)
        if goodput_match:
            results["request_goodput"] = float(goodput_match.group(1))

        return results

    def reset_prefix_cache(self):
        """Reset the prefix cache on the server."""
        reset_url = (
            f"http://{self.config.server.get_host()}:{self.config.server.port}"
            "/reset_prefix_cache"
        )
        try:
            requests.post(reset_url, timeout=5)
        except requests.RequestException as e:
            print(f"Warning: Failed to reset prefix cache: {e}")


def find_max_gpu_memory_utilization(
    config: TuneConfig, log_dir: Path, start_value: float = 0.95, min_value: float = 0.9
) -> float | None:
    """
    Find the maximum GPU memory utilization that doesn't cause OOM.

    Starts from start_value and decreases by 0.01 until the server starts successfully
    or reaches min_value.

    Args:
        config: Configuration
        log_dir: Directory for logs
        start_value: Starting GPU memory utilization
        min_value: Minimum GPU memory utilization to try

    Returns:
        GPU memory utilization value or None if none found
    """
    max_seqs = max(config.search_space.max_num_seqs)
    max_tokens = max(config.search_space.max_num_batched_tokens)

    gpu_mem = start_value
    while gpu_mem >= min_value:
        manager = ServerManager(
            config=config,
            log_dir=log_dir,
            gpu_memory_utilization=gpu_mem,
            max_num_seqs=max_seqs,
            max_num_batched_tokens=max_tokens,
        )
        if manager.start(timeout_seconds=300):
            manager.stop()
            print(f"Found working gpu_memory_utilization: {gpu_mem}")
            return gpu_mem
        manager.stop()
        gpu_mem -= 0.01
        gpu_mem = round(gpu_mem, 2)

    print(f"Could not find a working gpu_memory_utilization >= {min_value}")
    return None
