# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload handling for vLLM auto-tuning benchmarks."""

from config import WorkloadConfig


def build_dataset_args(workload: WorkloadConfig) -> list[str]:
    """
    Build command-line arguments for vllm bench serve based on workload configuration.

    Args:
        workload: Workload configuration

    Returns:
        List of command-line argument strings (e.g., ["--dataset-name", "random", ...])
    """
    args = []

    if workload.dataset_type == "random":
        # Calculate prefix length and adjusted input length
        prefix_len = (workload.input_len * workload.min_cache_hit_pct) // 100
        adjusted_input_len = workload.input_len - prefix_len

        args.extend(
            [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(adjusted_input_len),
                "--random-output-len",
                str(workload.output_len),
                "--random-prefix-len",
                str(prefix_len),
                "--seed",
                str(workload.seed),
            ]
        )
    elif workload.dataset_type == "custom":
        args.extend(
            [
                "--dataset-name",
                "custom",
                "--dataset-path",
                workload.dataset_path,
                "--custom-output-len",
                str(workload.output_len),
                "--seed",
                str(workload.seed),
            ]
        )
    else:
        raise ValueError(f"Unknown dataset_type: {workload.dataset_type}")

    # Add num-prompts if specified
    if workload.num_prompts:
        args.extend(["--num-prompts", str(workload.num_prompts)])

    # Handle oversample flag
    # Note: vllm bench serve uses --no-oversample flag to disable oversampling
    # By default it oversamples, so we only add the flag if oversample is False
    if not workload.oversample:
        args.append("--no-oversample")

    return args


def get_prefix_len(workload: WorkloadConfig) -> int:
    """
    Get the prefix length for the workload.

    For custom datasets, prefix caching simulation is disabled (returns 0).
    For random datasets, returns the calculated prefix length.
    """
    if workload.dataset_type == "custom":
        return 0
    return (workload.input_len * workload.min_cache_hit_pct) // 100


def get_adjusted_input_len(workload: WorkloadConfig) -> int:
    """
    Get the adjusted input length (input_len - prefix_len).

    For custom datasets, returns the full input_len.
    For random datasets, returns input_len - prefix_len.
    """
    if workload.dataset_type == "custom":
        return workload.input_len
    prefix_len = get_prefix_len(workload)
    return workload.input_len - prefix_len
