import csv
import json
import os
import sys
from typing import Optional, List, Dict, Any
import typer

from sage_quant.data import load_dataset, load_config, append_run
from sage_quant.models import Constraint
from sage_quant.recommender import recommend
from sage_quant.serve_config import generate_serving_config
from sage_quant.catalog import get_engine_note, get_algo_note

app = typer.Typer(help="SageQuant: Inference tradeoff calculator")


def parse_model_size(size_str: str) -> float:
    s = size_str.lower().strip()
    if s.startswith('-'):
        typer.echo(f"Error: Model size cannot be negative: '{size_str}'.", err=True)
        raise typer.Exit(code=1)
    if s.endswith('b'):
        s = s[:-1]
    try:
        value = float(s)
    except ValueError:
        typer.echo(f"Error: Invalid model size format '{size_str}'. Expected formats like '7b', '70b', or '7.0'.", err=True)
        raise typer.Exit(code=1)
    if value <= 0:
        typer.echo(f"Error: Model size must be greater than 0 (got '{size_str}').", err=True)
        raise typer.Exit(code=1)
    return value


def parse_latency(latency_str: Optional[str]) -> Optional[float]:
    if not latency_str:
        return None
    s = latency_str.lower().strip()
    if s.endswith('ms'):
        s = s[:-2]
    elif s.endswith('s'):
        try:
            val = float(s[:-1]) * 1000.0
            if val <= 0:
                typer.echo(f"Error: Latency must be greater than 0 (got '{latency_str}').", err=True)
                raise typer.Exit(code=1)
            return val
        except ValueError:
            pass
    try:
        val = float(s)
    except ValueError:
        typer.echo(f"Error: Invalid latency format '{latency_str}'. Expected formats like '200ms', '0.2s', or '200'.", err=True)
        raise typer.Exit(code=1)
    if val <= 0:
        typer.echo(f"Error: Latency must be greater than 0 (got '{latency_str}').", err=True)
        raise typer.Exit(code=1)
    return val


def resolve_user_dataset_path() -> str:
    """
    Always returns the user-writable dataset path at ~/.sage-quant/benchmarks.csv.
    Used by 'contribute' to avoid writing back into the pip-installed package.
    """
    return os.path.expanduser("~/.sage-quant/benchmarks.csv")


def resolve_dataset_path(config: Dict[str, Any]) -> str:
    """
    Resolve the dataset path for READ commands. Priority:
    1. Explicit dataset_path from config file.
    2. User's local dataset at ~/.sage-quant/benchmarks.csv (merged contributions).
    3. Bundled data/benchmarks.csv inside the package (survives pip install).
    4. Fallback: relative to the package directory (for dev/editable installs).
    """
    dataset_path = config.get("dataset_path")
    if dataset_path:
        return os.path.expanduser(dataset_path)

    # Prefer user-local dataset (includes contributed runs) if it exists
    user_path = resolve_user_dataset_path()
    if os.path.exists(user_path):
        return user_path

    # Use importlib.resources to find the bundled file after pip install
    try:
        import importlib.resources as pkg_resources
        ref = pkg_resources.files("sage_quant").joinpath("datasets/benchmarks.csv")
        with pkg_resources.as_file(ref) as p:
            if os.path.exists(str(p)):
                return str(p)
    except (AttributeError, TypeError, FileNotFoundError):
        pass

    # Fallback for editable/dev installs: file relative to package directory
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(pkg_dir, "datasets", "benchmarks.csv")
    if os.path.exists(candidate):
        return candidate

    # Last fallback: two levels up (repo root / data/)
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))
    return os.path.join(repo_root, "data", "benchmarks.csv")


def validate_tokens(prompt_tokens: int, output_tokens: int) -> None:
    if prompt_tokens < 0:
        typer.echo("Error: --prompt-tokens cannot be negative.", err=True)
        raise typer.Exit(code=1)
    if output_tokens < 0:
        typer.echo("Error: --output-tokens cannot be negative.", err=True)
        raise typer.Exit(code=1)


def validate_hardware(hw: str) -> str:
    hw = hw.strip()
    if not hw:
        typer.echo("Error: --hardware cannot be empty or whitespace.", err=True)
        raise typer.Exit(code=1)
    return hw.lower()


@app.command(name="recommend")
def recommend_cmd(
    model_size: str = typer.Option(..., "--model-size", "-m", help="Model size, e.g., 7b, 70b"),
    hardware: Optional[str] = typer.Option(None, "--hardware", "-hw", help="Hardware configuration, e.g., a100-40gb"),
    max_latency: Optional[str] = typer.Option(None, "--max-latency", "-l", help="Maximum latency constraint (checked against p95), e.g., 200ms"),
    min_quality: Optional[float] = typer.Option(None, "--min-quality", "-q", help="Minimum quality percentage, e.g., 97.0"),
    prompt_tokens: int = typer.Option(128, "--prompt-tokens", help="Workload shape prompt tokens"),
    output_tokens: int = typer.Option(128, "--output-tokens", help="Workload shape output tokens"),
    prefer_engine: Optional[str] = typer.Option(None, "--prefer-engine", "-pe", help="Preferred inference engine, e.g., sglang"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    Recommend an inference engine, quantization algorithm, and bit-width scheme.
    """
    config = load_config(config_path)

    final_hw = hardware or config.get("default_hardware")
    if not final_hw:
        typer.echo("Error: Hardware must be specified via --hardware or configured as 'default_hardware' in config.yaml.", err=True)
        raise typer.Exit(code=1)

    final_hw = validate_hardware(final_hw)
    validate_tokens(prompt_tokens, output_tokens)

    final_quality = min_quality
    if final_quality is None:
        final_quality = float(config.get("min_quality_default", 97.0))

    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        typer.echo("Tip: Run 'sage-quant list-hardware' after installing to confirm the bundled dataset is accessible.", err=True)
        raise typer.Exit(code=1)

    try:
        dataset = load_dataset(dataset_path)
    except (OSError, ValueError) as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1)

    parsed_model_size = parse_model_size(model_size)
    parsed_latency = parse_latency(max_latency)

    constraint = Constraint(
        model_size_b=parsed_model_size,
        hardware=final_hw,
        max_latency_ms=parsed_latency,
        min_quality_pct=final_quality,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefer_engine=prefer_engine
    )

    rec = recommend(constraint, dataset)
    if rec is None:
        typer.echo("no matching data — try a different hardware or looser quality bound", err=True)
        raise typer.Exit(code=1)

    num_runs = len(rec.source_rows)
    runs_str = "run" if num_runs == 1 else "runs"

    algo_display = rec.quant_algo.upper() if rec.quant_algo != "none" else "none"
    cache_display = "on" if rec.prefix_caching else "off"
    engine_name = rec.inference_engine.upper() if rec.inference_engine != 'mlx' else 'MLX'

    typer.echo(f"Recommended: {engine_name} + {algo_display} ({rec.quant_scheme}), prefix caching {cache_display}")
    typer.echo(f"Workload: {constraint.prompt_tokens} in / {constraint.output_tokens} out tokens")
    typer.echo(f"Expected: {rec.expected_ttft_p50_ms:.0f}ms TTFT (p50) · {rec.expected_ttft_p95_ms:.0f}ms TTFT (p95) · {rec.expected_throughput:.1f} tok/s · {rec.quality_delta_pct:+.1f}% quality ({rec.eval_method}, n={rec.eval_sample_size})")
    typer.echo(f"Confidence: {rec.confidence} ({num_runs} matching benchmark {runs_str})")

    engine_note = get_engine_note(rec.inference_engine)
    algo_note = get_algo_note(rec.quant_algo)
    if engine_note or algo_note:
        typer.echo("")
        if engine_note:
            typer.echo(f"Note (Engine): {engine_note}")
        if algo_note:
            typer.echo(f"Note (Algo): {algo_note}")


@app.command(name="serve-config")
def serve_config_cmd(
    model_size: str = typer.Option(..., "--model-size", "-m", help="Model size, e.g., 7b, 70b"),
    hardware: Optional[str] = typer.Option(None, "--hardware", "-hw", help="Hardware configuration, e.g., a100-40gb"),
    model_name: str = typer.Option(..., "--model", help="HuggingFace model name, e.g., meta-llama/Meta-Llama-3-8B-Instruct"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file path for YAML config"),
    min_quality: Optional[float] = typer.Option(None, "--min-quality", "-q", help="Minimum quality percentage"),
    max_latency: Optional[str] = typer.Option(None, "--max-latency", "-l", help="Maximum latency constraint, e.g., 200ms"),
    prompt_tokens: int = typer.Option(128, "--prompt-tokens", help="Workload shape prompt tokens"),
    output_tokens: int = typer.Option(128, "--output-tokens", help="Workload shape output tokens"),
    prefer_engine: Optional[str] = typer.Option(None, "--prefer-engine", "-pe", help="Preferred inference engine"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    Generate a runnable serving config from the recommendation.
    """
    config = load_config(config_path)
    final_hw = hardware or config.get("default_hardware")
    if not final_hw:
        typer.echo("Error: Hardware must be specified via --hardware or configured as 'default_hardware' in config.yaml.", err=True)
        raise typer.Exit(code=1)

    final_hw = validate_hardware(final_hw)
    validate_tokens(prompt_tokens, output_tokens)

    final_quality = min_quality if min_quality is not None else float(config.get("min_quality_default", 97.0))
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)

    try:
        dataset = load_dataset(dataset_path)
    except (OSError, ValueError) as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1)

    parsed_model_size = parse_model_size(model_size)
    parsed_latency = parse_latency(max_latency)

    constraint = Constraint(
        model_size_b=parsed_model_size,
        hardware=final_hw,
        max_latency_ms=parsed_latency,
        min_quality_pct=final_quality,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefer_engine=prefer_engine
    )

    rec = recommend(constraint, dataset)
    if rec is None:
        typer.echo("no matching data — try a different hardware or looser quality bound", err=True)
        raise typer.Exit(code=1)

    cfg = generate_serving_config(rec.inference_engine, rec.quant_algo, rec.quant_scheme, model_name)

    algo_display = rec.quant_algo.upper() if rec.quant_algo != "none" else "none"
    cache_display = "on" if rec.prefix_caching else "off"
    engine_name = rec.inference_engine.upper() if rec.inference_engine != 'mlx' else 'MLX'

    typer.echo(f"")
    typer.echo(f"Recommended: {engine_name} + {algo_display} ({rec.quant_scheme}), prefix caching {cache_display}")
    typer.echo(f"Workload: {constraint.prompt_tokens} in / {constraint.output_tokens} out tokens")
    typer.echo(f"Expected: {rec.expected_ttft_p50_ms:.0f}ms TTFT (p50) · {rec.expected_ttft_p95_ms:.0f}ms TTFT (p95) · {rec.expected_throughput:.1f} tok/s · {rec.quality_delta_pct:+.1f}% quality ({rec.eval_method}, n={rec.eval_sample_size})")
    typer.echo(f"Confidence: {rec.confidence}")
    typer.echo(f"")
    typer.echo(f"Platform: {cfg['platform']}")
    typer.echo(f"")
    typer.echo("Launch command:")
    typer.echo(f"  {cfg['launch_command']}")
    typer.echo(f"")

    if cfg["platform"] == "vllm" and "yaml_content" in cfg:
        if out:
            with open(out, "w") as f:
                f.write(cfg["yaml_content"])
            typer.echo(f"Config written to: {out}")
        else:
            typer.echo("Config YAML:")
            for line in cfg["yaml_content"].splitlines():
                typer.echo(f"  {line}")
        typer.echo("")

    typer.echo("Next steps:")
    for step in cfg["instructions"]:
        typer.echo(f"  {step}")


@app.command(name="list-hardware")
def list_hardware_cmd(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    List unique hardware configurations in the dataset.
    """
    config = load_config(config_path)
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)

    try:
        dataset = load_dataset(dataset_path)
    except (OSError, ValueError) as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1)

    unique_hw = sorted(list(set(row["hardware"] for row in dataset)))
    for hw in unique_hw:
        typer.echo(hw)


@app.command(name="list-engines")
def list_engines_cmd(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    List unique inference engines in the dataset.
    """
    config = load_config(config_path)
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)

    try:
        dataset = load_dataset(dataset_path)
    except (OSError, ValueError) as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1)

    unique_engines = sorted(list(set(row["inference_engine"] for row in dataset)))
    for engine in unique_engines:
        typer.echo(engine)


@app.command(name="list-quant-algos")
def list_quant_algos_cmd(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    List unique quantization algorithms in the dataset.
    """
    config = load_config(config_path)
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)

    try:
        dataset = load_dataset(dataset_path)
    except (OSError, ValueError) as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1)

    unique_algos = sorted(list(set(row["quant_algo"] for row in dataset)))
    for algo in unique_algos:
        typer.echo(algo)


@app.command(name="contribute")
def contribute_cmd(
    run_log: Optional[str] = typer.Option(None, "--run-log", "-r", help="Path to JSON or CSV run log"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    Contribute a benchmark run log to your local dataset and get sharing instructions.

    Accepts a JSON file (single dict or list of dicts) or CSV file.
    Each row must include all required benchmark fields.

    To share your results back with the community, open a PR to the repository
    after appending your data.
    """
    if not run_log:
        typer.echo("Error: --run-log is required. Provide a JSON or CSV file with your benchmark data.", err=True)
        raise typer.Exit(code=1)

    config = load_config(config_path)
    # IMPORTANT: contribute always writes to the user's local path (~/.sage-quant/)
    # NOT to the bundled dataset inside the pip install directory.
    dataset_path = config.get("dataset_path") or resolve_user_dataset_path()
    dataset_path = os.path.expanduser(dataset_path)

    if not os.path.exists(run_log):
        typer.echo(f"Error: Run log file not found at {run_log}", err=True)
        raise typer.Exit(code=1)

    try:
        if run_log.endswith(".csv"):
            with open(run_log, "r", encoding="utf-8") as f:
                runs = list(csv.DictReader(f))
        else:
            with open(run_log, "r", encoding="utf-8") as f:
                data = json.load(f)
                runs = data if isinstance(data, list) else [data]
    except Exception as e:
        typer.echo(f"Error parsing run log: {e}", err=True)
        raise typer.Exit(code=1)

    if not runs:
        typer.echo("Error: No runs to contribute", err=True)
        raise typer.Exit(code=1)

    success_count = 0
    for idx, r in enumerate(runs):
        try:
            append_run(dataset_path, r)
            success_count += 1
        except Exception as e:
            typer.echo(f"Error validating run #{idx + 1}: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"Successfully appended {success_count} benchmark run(s) to the dataset at {dataset_path}!")
    typer.echo("")
    typer.echo("To share your results back with the community:")
    typer.echo("1. Fork the repository: https://github.com/aakriti1318/sage-quant")
    typer.echo("2. Create a new branch: git checkout -b add-my-benchmarks")
    typer.echo("3. Copy your additions to data/benchmarks.csv")
    typer.echo("4. Commit and push: git push origin add-my-benchmarks")
    typer.echo("5. Open a Pull Request on GitHub.")


def main():
    app()


if __name__ == "__main__":
    main()
