# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration loading and validation for vLLM auto-tuning."""

import socket
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str
    tensor_parallel_size: int = 1
    download_dir: str = ""
    max_model_len: int = 4096

    def __post_init__(self):
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str | None = None
    port: int = 8004
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool = True
    load_format: str = "dummy"
    disable_log_requests: bool = True

    def __post_init__(self):
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        if self.gpu_memory_utilization is not None and not (
            0.0 < self.gpu_memory_utilization <= 1.0
        ):
            raise ValueError(
                "gpu_memory_utilization must be between 0 and 1, "
                f"got {self.gpu_memory_utilization}"
            )

    def get_host(self) -> str:
        """Get the hostname, auto-detecting if None."""
        if self.host is None:
            return socket.gethostname()
        return self.host


@dataclass
class WorkloadConfig:
    """Workload configuration."""

    dataset_type: str = "random"
    input_len: int = 4000
    output_len: int = -1
    min_cache_hit_pct: int = 0
    dataset_path: str = "data.jsonl"
    seed: int = 42
    num_prompts: int = 1000
    oversample: bool = True

    def __post_init__(self):
        if self.dataset_type not in ("random", "custom"):
            raise ValueError(
                f"dataset_type must be 'random' or 'custom', got {self.dataset_type}"
            )
        if self.input_len < 1:
            raise ValueError(f"input_len must be >= 1, got {self.input_len}")
        if self.min_cache_hit_pct < 0 or self.min_cache_hit_pct > 100:
            raise ValueError(
                "min_cache_hit_pct must be between 0 and 100, "
                f"got {self.min_cache_hit_pct}"
            )
        if self.dataset_type == "custom" and not self.dataset_path:
            raise ValueError(
                "dataset_path must be provided when dataset_type is 'custom'"
            )
        if self.num_prompts < 1:
            raise ValueError(f"num_prompts must be >= 1, got {self.num_prompts}")


@dataclass
class SearchSpaceConfig:
    """Search space configuration."""

    max_num_seqs: list[int] = field(default_factory=lambda: [128])
    max_num_batched_tokens: list[int] = field(default_factory=lambda: [2048, 4096])
    # Placeholder for future speculative decoding
    # speculative_decoding: Optional[Dict[str, Any]] = None
    # Placeholder for future length sampler
    # length_sampler: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.max_num_seqs:
            raise ValueError("max_num_seqs cannot be empty")
        if not self.max_num_batched_tokens:
            raise ValueError("max_num_batched_tokens cannot be empty")
        if any(x < 1 for x in self.max_num_seqs):
            raise ValueError("All max_num_seqs values must be >= 1")
        if any(x < 1 for x in self.max_num_batched_tokens):
            raise ValueError("All max_num_batched_tokens values must be >= 1")


@dataclass
class ConstraintsConfig:
    """Performance constraints."""

    max_p99_e2el_ms: float = 100000000000.0
    # Future: min_throughput, max_memory_usage, etc.

    def __post_init__(self):
        if self.max_p99_e2el_ms < 0:
            raise ValueError(
                f"max_p99_e2el_ms must be >= 0, got {self.max_p99_e2el_ms}"
            )


@dataclass
class StrategyConfig:
    """Search strategy configuration."""

    type: str = "grid"

    def __post_init__(self):
        if self.type != "grid":
            raise ValueError(f"strategy.type must be 'grid', got {self.type}")


@dataclass
class ObjectiveConfig:
    """Optimization objective."""

    metric: str = "throughput"
    direction: str = "maximize"

    def __post_init__(self):
        if self.metric not in ("throughput", "goodput"):
            raise ValueError(
                f"objective.metric must be 'throughput' or 'goodput', got {self.metric}"
            )
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                "objective.direction must be 'maximize' or 'minimize', "
                f"got {self.direction}"
            )


@dataclass
class RunConfig:
    """Run control configuration."""

    request_rate_mode: str = "auto"
    request_rate: float | None = None
    percentile_metrics: list[str] = field(
        default_factory=lambda: ["ttft", "tpot", "itl", "e2el"]
    )
    disable_tqdm: bool = True
    ignore_eos: bool = True
    enable_profiling: bool = False
    profile_dir: str | None = None

    def __post_init__(self):
        if self.request_rate_mode not in ("auto", "fixed"):
            raise ValueError(
                "request_rate_mode must be 'auto' or 'fixed', "
                f"got {self.request_rate_mode}"
            )
        if self.request_rate_mode == "fixed" and self.request_rate is None:
            raise ValueError(
                "request_rate must be specified when request_rate_mode is 'fixed'"
            )
        valid_metrics = {"ttft", "tpot", "itl", "e2el"}
        invalid = set(self.percentile_metrics) - valid_metrics
        if invalid:
            raise ValueError(
                f"Invalid percentile_metrics: {invalid}. Valid options: {valid_metrics}"
            )


@dataclass
class OutputConfig:
    """Output configuration."""

    base_dir: str | None = None
    results_file: str = "results.jsonl"


@dataclass
class TuneConfig:
    """Complete auto-tuning configuration."""

    model: ModelConfig
    server: ServerConfig
    workload: WorkloadConfig
    search_space: SearchSpaceConfig
    constraints: ConstraintsConfig
    strategy: StrategyConfig
    objective: ObjectiveConfig
    run: RunConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TuneConfig":
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML file must contain a dictionary")

        # Validate that input_len + output_len <= max_model_len
        workload_data = data.get("workload", {})
        model_data = data.get("model", {})
        input_len = workload_data.get("input_len", 4000)
        output_len = workload_data.get("output_len", -1)
        max_model_len = model_data.get("max_model_len", 4096)

        if output_len > 0:
            total_len = input_len + output_len
            if total_len > max_model_len:
                raise ValueError(
                    f"input_len ({input_len}) + output_len ({output_len}) "
                    f"= {total_len}, which exceeds max_model_len "
                    f"({max_model_len})"
                )

        return cls(
            model=ModelConfig(**data.get("model", {})),
            server=ServerConfig(**data.get("server", {})),
            workload=WorkloadConfig(**data.get("workload", {})),
            search_space=SearchSpaceConfig(**data.get("search_space", {})),
            constraints=ConstraintsConfig(**data.get("constraints", {})),
            strategy=StrategyConfig(**data.get("strategy", {})),
            objective=ObjectiveConfig(**data.get("objective", {})),
            run=RunConfig(**data.get("run", {})),
            output=OutputConfig(**data.get("output", {})),
        )

    def validate(self) -> None:
        """Validate the complete configuration."""
        # Additional cross-field validations can go here
        self.model.__post_init__()
        self.server.__post_init__()
        self.workload.__post_init__()
        self.search_space.__post_init__()
        self.constraints.__post_init__()
        self.strategy.__post_init__()
        self.objective.__post_init__()
        self.run.__post_init__()
