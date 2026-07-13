import csv
import os
import importlib.resources
from typing import List, Dict, Any, Optional


def str_to_bool(val: str) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a benchmarks CSV dataset. Raises clear errors on malformed input."""
    dataset = []
    try:
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {
                "model_size_b", "hardware", "inference_engine", "quant_algo",
                "quant_scheme", "prompt_tokens", "output_tokens", "prefix_caching",
                "ttft_p50_ms", "ttft_p95_ms", "throughput_tok_s", "perplexity_delta",
                "task_score_delta", "eval_method", "eval_sample_size", "vram_gb", "source"
            }
            headers = set(reader.fieldnames or [])
            missing = required_cols - headers
            if missing:
                raise ValueError(f"Dataset CSV is missing required columns: {', '.join(sorted(missing))}")
            for i, row in enumerate(reader, start=2):  # row 1 is header
                try:
                    dataset.append({
                        "model_size_b": float(row["model_size_b"]),
                        "hardware": row["hardware"].strip().lower(),
                        "inference_engine": row["inference_engine"].strip().lower(),
                        "quant_algo": row["quant_algo"].strip().lower(),
                        "quant_scheme": row["quant_scheme"].strip(),
                        "prompt_tokens": int(row["prompt_tokens"]),
                        "output_tokens": int(row["output_tokens"]),
                        "prefix_caching": str_to_bool(row["prefix_caching"]),
                        "ttft_p50_ms": float(row["ttft_p50_ms"]),
                        "ttft_p95_ms": float(row["ttft_p95_ms"]),
                        "throughput_tok_s": float(row["throughput_tok_s"]),
                        "perplexity_delta": float(row["perplexity_delta"]),
                        "task_score_delta": float(row["task_score_delta"]),
                        "eval_method": row["eval_method"].strip(),
                        "eval_sample_size": int(row["eval_sample_size"]),
                        "vram_gb": float(row["vram_gb"]),
                        "source": row["source"].strip()
                    })
                except (KeyError, ValueError) as e:
                    raise ValueError(f"Dataset CSV row {i} is malformed: {e}")
    except OSError as e:
        raise OSError(f"Cannot read dataset file '{path}': {e}")
    return dataset


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.expanduser("~/.sage_quant/config.yaml")
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (ImportError, Exception):
        return {}


def append_run(dataset_path: str, new_run: Dict[str, Any]) -> None:
    required_fields = [
        "model_size_b", "hardware", "inference_engine", "quant_algo", "quant_scheme",
        "prompt_tokens", "output_tokens", "prefix_caching", "ttft_p50_ms", "ttft_p95_ms",
        "throughput_tok_s", "perplexity_delta", "task_score_delta", "eval_method",
        "eval_sample_size", "vram_gb", "source"
    ]

    for field in required_fields:
        if field not in new_run:
            raise ValueError(f"Missing required field: '{field}'")

    # Validate model_size_b separately so the real message isn't swallowed
    try:
        model_size_b = float(new_run["model_size_b"])
    except (TypeError, ValueError):
        raise ValueError("model_size_b must be a valid positive number")
    if model_size_b <= 0:
        raise ValueError("model_size_b must be greater than 0")

    hardware = str(new_run["hardware"]).strip()
    if not hardware:
        raise ValueError("hardware cannot be empty")

    inference_engine = str(new_run["inference_engine"]).strip()
    if not inference_engine:
        raise ValueError("inference_engine cannot be empty")

    quant_algo = str(new_run["quant_algo"]).strip()
    if not quant_algo:
        raise ValueError("quant_algo cannot be empty")

    quant_scheme = str(new_run["quant_scheme"]).strip()
    if not quant_scheme:
        raise ValueError("quant_scheme cannot be empty")

    try:
        prompt_tokens = int(new_run["prompt_tokens"])
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens cannot be negative")
    except (TypeError, ValueError):
        raise ValueError("prompt_tokens must be a non-negative integer")

    try:
        output_tokens = int(new_run["output_tokens"])
        if output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")
    except (TypeError, ValueError):
        raise ValueError("output_tokens must be a non-negative integer")

    prefix_caching = str_to_bool(new_run["prefix_caching"])

    try:
        ttft_p50_ms = float(new_run["ttft_p50_ms"])
        if ttft_p50_ms < 0:
            raise ValueError("ttft_p50_ms cannot be negative")
    except (TypeError, ValueError):
        raise ValueError("ttft_p50_ms must be a valid non-negative number")

    try:
        ttft_p95_ms = float(new_run["ttft_p95_ms"])
        if ttft_p95_ms < 0:
            raise ValueError("ttft_p95_ms cannot be negative")
    except (TypeError, ValueError):
        raise ValueError("ttft_p95_ms must be a valid non-negative number")

    try:
        throughput_tok_s = float(new_run["throughput_tok_s"])
        if throughput_tok_s < 0:
            raise ValueError("throughput_tok_s cannot be negative")
    except (TypeError, ValueError):
        raise ValueError("throughput_tok_s must be a valid non-negative number")

    try:
        perplexity_delta = float(new_run["perplexity_delta"])
    except (TypeError, ValueError):
        raise ValueError("perplexity_delta must be a valid number")

    try:
        task_score_delta = float(new_run["task_score_delta"])
    except (TypeError, ValueError):
        raise ValueError("task_score_delta must be a valid number")

    eval_method = str(new_run["eval_method"]).strip()
    if not eval_method:
        raise ValueError("eval_method cannot be empty")

    try:
        eval_sample_size = int(new_run["eval_sample_size"])
        if eval_sample_size < 0:
            raise ValueError("eval_sample_size cannot be negative")
    except (TypeError, ValueError):
        raise ValueError("eval_sample_size must be a non-negative integer")

    try:
        vram_gb = float(new_run["vram_gb"])
        if vram_gb < 0:
            raise ValueError("vram_gb cannot be negative")
    except (TypeError, ValueError):
        raise ValueError("vram_gb must be a valid non-negative number")

    source = str(new_run["source"]).strip()
    if not source:
        raise ValueError("source cannot be empty")

    row_to_append = {
        "model_size_b": model_size_b,
        "hardware": hardware.lower(),
        "inference_engine": inference_engine.lower(),
        "quant_algo": quant_algo.lower(),
        "quant_scheme": quant_scheme,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefix_caching": "true" if prefix_caching else "false",
        "ttft_p50_ms": ttft_p50_ms,
        "ttft_p95_ms": ttft_p95_ms,
        "throughput_tok_s": throughput_tok_s,
        "perplexity_delta": perplexity_delta,
        "task_score_delta": task_score_delta,
        "eval_method": eval_method,
        "eval_sample_size": eval_sample_size,
        "vram_gb": vram_gb,
        "source": source
    }

    file_exists = os.path.exists(dataset_path)
    dir_name = os.path.dirname(dataset_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(dataset_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=required_fields)
        if not file_exists or os.path.getsize(dataset_path) == 0:
            writer.writeheader()
        writer.writerow(row_to_append)
