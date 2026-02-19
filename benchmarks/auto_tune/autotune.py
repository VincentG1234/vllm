#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main entrypoint for vLLM auto-tuning."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from config import TuneConfig
from runner import BenchmarkRunner, ServerManager, find_max_gpu_memory_utilization
from workload import get_prefix_len


def generate_config_id(
    max_num_seqs: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> str:
    """Generate a stable config ID string."""
    return (
        f"seq{max_num_seqs}_tok{max_num_batched_tokens}_"
        f"gmu{gpu_memory_utilization:.2f}_tp{tensor_parallel_size}"
    )


def generate_search_space(config: TuneConfig) -> list[dict]:
    """
    Generate all configurations to test based on search space.

    Currently only supports grid search.
    """
    configs = []
    for max_num_seqs in config.search_space.max_num_seqs:
        for max_num_batched_tokens in config.search_space.max_num_batched_tokens:
            configs.append(
                {
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": max_num_batched_tokens,
                }
            )
    return configs


def run_single_config(
    config: TuneConfig,
    run_dir: Path,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> dict:
    """
    Run a single configuration and return results.

    Returns:
        Dictionary with config, metrics, status, and log paths
    """
    config_id = generate_config_id(
        max_num_seqs,
        max_num_batched_tokens,
        gpu_memory_utilization,
        config.model.tensor_parallel_size,
    )
    config_dir = run_dir / config_id
    config_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "config_id": config_id,
        "config": {
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": config.model.tensor_parallel_size,
        },
        "status": "failed",
        "logs": {
            "server_log": str(config_dir / "server.log"),
            "bench_log": str(config_dir / "bench.log"),
        },
        "metrics": {},
    }

    prefix_len = get_prefix_len(config.workload)

    # Start server
    manager = ServerManager(
        config=config,
        log_dir=config_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    if not manager.start():
        result["error"] = "Server failed to start"
        return result

    try:
        # Run benchmark
        runner = BenchmarkRunner(config, config_dir)

        # First, try with request_rate=inf to get baseline throughput
        print(
            f"Running benchmark with request_rate=inf for "
            f"max_num_seqs={max_num_seqs}, "
            f"max_num_batched_tokens={max_num_batched_tokens}"
        )
        metrics, success = runner.run(
            request_rate=float("inf"),
            prefix_len=prefix_len,
            use_oversample=True,
        )

        if not success:
            result["error"] = "Benchmark failed"
            return result

        request_rate = float("inf")
        e2el_ms = metrics.get("p99_e2el_ms", float("inf"))

        # Check if latency constraint is met
        if e2el_ms > config.constraints.max_p99_e2el_ms:
            # Need to reduce request rate
            print(
                f"P99 E2EL ({e2el_ms} ms) exceeds constraint "
                f"({config.constraints.max_p99_e2el_ms} ms). Reducing request rate..."
            )
            throughput = metrics.get("request_throughput", 0)
            request_rate = int(throughput) + 1

            # Binary search down until constraint is met
            while request_rate > 0:
                print(f"Trying request_rate={request_rate}")
                runner.reset_prefix_cache()
                time.sleep(5)  # Wait for cache reset

                metrics, success = runner.run(
                    request_rate=request_rate,
                    prefix_len=prefix_len,
                    use_oversample=False,  # Don't oversample when tuning request rate
                )

                if not success:
                    request_rate -= 1
                    continue

                e2el_ms = metrics.get("p99_e2el_ms", float("inf"))
                if e2el_ms <= config.constraints.max_p99_e2el_ms:
                    print(
                        f"Found working request_rate={request_rate} "
                        f"with E2EL={e2el_ms} ms"
                    )
                    break

                request_rate -= 1

            if request_rate <= 0:
                result["error"] = (
                    "Could not find request_rate that meets latency constraint"
                )
                return result

        # Save result JSON if available
        result_json_path = config_dir / "result.json"
        if "request_throughput" in metrics:
            with open(result_json_path, "w") as f:
                json.dump(metrics, f, indent=2)

        result["status"] = "success"
        result["metrics"] = metrics
        result["metrics"]["request_rate"] = (
            request_rate if request_rate != float("inf") else "inf"
        )
        result["result_json"] = str(result_json_path)

        print(
            f"Config {config_id}: throughput="
            f"{metrics.get('request_throughput', 'N/A')}, "
            f"goodput={metrics.get('request_goodput', 'N/A')}, "
            f"E2EL={metrics.get('p99_e2el_ms', 'N/A')} ms"
        )

    finally:
        manager.stop()

    return result


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="vLLM auto-tuning tool")
    parser.add_argument(
        "--config",
        type=str,
        default="tune.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = TuneConfig.from_yaml(args.config)
        config.validate()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(config.output.base_dir) if config.output.base_dir else Path("runs")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration to run directory
    config_path = run_dir / "config.yaml"
    with open(args.config) as f:
        config_data = yaml.safe_load(f)
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    print(f"Starting auto-tuning run: {run_dir}")
    print(f"Configuration: {args.config}")

    # Find maximum GPU memory utilization
    print("Finding maximum GPU memory utilization...")
    gpu_memory_utilization = config.server.gpu_memory_utilization
    if gpu_memory_utilization is None:
        gpu_memory_utilization = find_max_gpu_memory_utilization(
            config, run_dir / "gpu_mem_probe"
        )
        if gpu_memory_utilization is None:
            print(
                "Error: Could not find working GPU memory utilization", file=sys.stderr
            )
            sys.exit(1)
    print(f"Using gpu_memory_utilization={gpu_memory_utilization}")

    # Generate search space
    configs_to_test = generate_search_space(config)
    print(f"Testing {len(configs_to_test)} configurations...")

    # Run each configuration
    results = []
    best_result = None
    best_score = (
        float("-inf") if config.objective.direction == "maximize" else float("inf")
    )

    for i, test_config in enumerate(configs_to_test):
        print(f"\n[{i + 1}/{len(configs_to_test)}] Testing configuration...")
        result = run_single_config(
            config=config,
            run_dir=run_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=test_config["max_num_seqs"],
            max_num_batched_tokens=test_config["max_num_batched_tokens"],
        )
        results.append(result)

        # Track best result
        if result["status"] == "success":
            metrics = result["metrics"]
            if config.objective.metric == "throughput":
                score = metrics.get("request_throughput", 0)
            else:  # goodput
                score = metrics.get("request_goodput", 0)

            if (config.objective.direction == "maximize" and score > best_score) or (
                config.objective.direction == "minimize" and score < best_score
            ):
                best_score = score
                best_result = result

    # Save results to JSONL
    results_file = run_dir / config.output.results_file
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if best_result:
        print(f"\nBest configuration: {best_result['config_id']}")
        thr = best_result["metrics"].get("request_throughput", "N/A")
        print(f"  Throughput: {thr} req/s")
        gput = best_result["metrics"].get("request_goodput", "N/A")
        print(f"  Goodput: {gput} req/s")
        print(f"  P99 E2EL: {best_result['metrics'].get('p99_e2el_ms', 'N/A')} ms")

    # Optional: Run profiling on best config
    if config.run.enable_profiling and best_result:
        print("\n" + "=" * 60)
        print("Running profiling on best configuration...")
        print("=" * 60)

        profile_dir = run_dir / "profile"
        if config.run.profile_dir:
            profile_dir = Path(config.run.profile_dir)

        config_id = best_result["config_id"]
        config_dir = run_dir / config_id

        manager = ServerManager(
            config=config,
            log_dir=config_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=best_result["config"]["max_num_seqs"],
            max_num_batched_tokens=best_result["config"]["max_num_batched_tokens"],
            profile_dir=profile_dir,
        )

        if manager.start():
            try:
                runner = BenchmarkRunner(config, config_dir)
                request_rate = best_result["metrics"].get("request_rate", "inf")
                if request_rate == "inf":
                    request_rate = float("inf")
                else:
                    request_rate = float(request_rate)

                runner.run(
                    request_rate=request_rate,
                    prefix_len=get_prefix_len(config.workload),
                    use_oversample=False,
                    profile=True,
                )
                print(f"Profiling complete. Results saved to: {profile_dir}")
            finally:
                manager.stop()

    print(f"\nRun complete. All results in: {run_dir}")


if __name__ == "__main__":
    main()
