from collections import defaultdict
from typing import List, Dict, Any, Optional
from sage_quant.models import Constraint, Recommendation

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

QUANT_BIT_WIDTHS = {
    "fp16": 16.0,
    "bf16": 16.0,
    "w8a8": 8.0,
    "q8": 8.0,
    "w4a16": 4.0,
    "q4": 4.0,
}


def get_bit_width(scheme: str) -> float:
    scheme_lower = scheme.lower()
    for key, width in QUANT_BIT_WIDTHS.items():
        if key in scheme_lower:
            return width
    if "16" in scheme_lower:
        return 16.0
    if "8" in scheme_lower:
        return 8.0
    if "4" in scheme_lower:
        return 4.0
    return 8.0


def score_candidate(p: Dict[str, Any]) -> tuple:
    """
    Multi-axis scoring: balance throughput, latency, and quality.

    Sort priority (higher is better):
    1. Quality delta (higher = less quality loss vs fp16) — filtered already to min_quality
    2. Effective score: throughput / latency_p95 ratio (throughput efficiency)
    3. Lower p95 latency (tiebreaker)

    This avoids the quality-uber-alles problem: once all candidates satisfy the
    quality floor, we rank by throughput-per-unit-latency so a faster quantized
    option beats a slower unquantized one.
    """
    quality = p["task_score_delta"]             # ≤ 0; higher is better (closer to fp16)
    latency = max(p["ttft_p95_ms"], 0.1)        # avoid /0
    efficiency = p["throughput_tok_s"] / latency # higher = better
    neg_latency = -latency                       # lower latency wins ties
    return (quality, efficiency, neg_latency)


def linear_fit_and_predict(x_pts: List[float], y_pts: List[float], target_x: float) -> float:
    n = len(x_pts)
    if n == 0:
        return 0.0
    if n == 1:
        return y_pts[0]
    mean_x = sum(x_pts) / n
    mean_y = sum(y_pts) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_pts, y_pts))
    den = sum((x - mean_x) ** 2 for x in x_pts)
    if den == 0:
        return mean_y
    m = num / den
    c = mean_y - m * mean_x
    return m * target_x + c


def predict_metric_for_model_size(runs: List[dict], target_size: float, metric_key: str) -> float:
    if not runs:
        return 0.0
    x_pts = [r["model_size_b"] for r in runs]
    y_pts = [r[metric_key] for r in runs]

    if len(set(x_pts)) == 1:
        val = y_pts[0]
        base_size = x_pts[0]
        if base_size <= 0:
            return val
        if metric_key in ("ttft_p50_ms", "ttft_p95_ms", "vram_gb"):
            return val * (target_size / base_size)
        elif metric_key == "throughput_tok_s":
            # Larger model → lower throughput; guard against zero target
            if target_size <= 0:
                return val
            return val * (base_size / target_size)
        else:
            return val

    return linear_fit_and_predict(x_pts, y_pts, target_size)


def filter_by_closest_workload(runs: List[Dict[str, Any]], target_prompt: int, target_output: int) -> List[Dict[str, Any]]:
    if not runs:
        return []
    min_dist = min(
        abs(r["prompt_tokens"] - target_prompt) + abs(r["output_tokens"] - target_output)
        for r in runs
    )
    return [
        r for r in runs
        if (abs(r["prompt_tokens"] - target_prompt) + abs(r["output_tokens"] - target_output)) == min_dist
    ]


def interpolate(constraint: Constraint, dataset: List[Dict[str, Any]]) -> Optional[Recommendation]:
    # 1. Filter by hardware — exact match first, then exact-family prefix (e.g. "a100")
    hw_lower = constraint.hardware.strip().lower()
    exact_hw_runs = [row for row in dataset if row["hardware"] == hw_lower]
    if exact_hw_runs:
        family_runs = exact_hw_runs
        hardware_fallback = False
    else:
        # Extract first token before dash as the family identifier (e.g. "a100" from "a100-80gb")
        parts = hw_lower.split('-')
        family = parts[0] if parts else hw_lower
        family_runs = [row for row in dataset if row["hardware"].startswith(family + "-") or row["hardware"] == family]
        hardware_fallback = bool(family_runs)
        if not family_runs:
            # Last resort: whole dataset but warn via confidence label
            family_runs = dataset
            hardware_fallback = False  # will be flagged as no-data confidence

    if not family_runs:
        return None

    # Match on the closest available workload shape first
    family_runs = filter_by_closest_workload(family_runs, constraint.prompt_tokens, constraint.output_tokens)

    # Filter by preferred engine if specified
    if constraint.prefer_engine:
        pref = constraint.prefer_engine.strip().lower()
        family_runs = [r for r in family_runs if r["inference_engine"] == pref]

    if not family_runs:
        return None

    # Group family runs by engine
    engine_runs = defaultdict(list)
    for r in family_runs:
        engine_runs[r["inference_engine"]].append(r)

    points = []

    for engine, runs in engine_runs.items():
        is_apple = (engine == "mlx")
        standard_configs = [
            ("mlx-quant", "q4", 4.0), ("mlx-quant", "q8", 8.0), ("none", "fp16", 16.0)
        ] if is_apple else [
            ("awq", "w4a16", 4.0), ("none", "w8a8", 8.0), ("none", "fp16", 16.0)
        ]

        # Group runs for this engine by quant_scheme
        scheme_runs = defaultdict(list)
        for r in runs:
            scheme_runs[r["quant_scheme"]].append(r)

        # Predict metrics for target model size for the schemes we have data for
        known_points = []
        for scheme, s_runs in scheme_runs.items():
            ttft_p50 = predict_metric_for_model_size(s_runs, constraint.model_size_b, "ttft_p50_ms")
            ttft_p95 = predict_metric_for_model_size(s_runs, constraint.model_size_b, "ttft_p95_ms")
            thru = predict_metric_for_model_size(s_runs, constraint.model_size_b, "throughput_tok_s")
            qual = predict_metric_for_model_size(s_runs, constraint.model_size_b, "task_score_delta")
            eval_method = s_runs[0]["eval_method"]
            eval_sample_size = s_runs[0]["eval_sample_size"]
            prefix_caching = s_runs[0]["prefix_caching"]
            quant_algo = s_runs[0]["quant_algo"]

            known_points.append({
                "inference_engine": engine,
                "quant_algo": quant_algo,
                "quant_scheme": scheme,
                "bit_width": get_bit_width(scheme),
                "ttft_p50_ms": max(0.1, ttft_p50),
                "ttft_p95_ms": max(0.1, ttft_p95),
                "throughput_tok_s": max(0.1, thru),
                "task_score_delta": qual,
                "eval_method": eval_method,
                "eval_sample_size": eval_sample_size,
                "prefix_caching": prefix_caching,
                "source_runs": s_runs
            })

        # Interpolate the standard configurations that are missing
        x_bits = [p["bit_width"] for p in known_points]
        if len(x_bits) > 0:
            for target_algo, target_scheme, target_bit in standard_configs:
                if any(p["quant_scheme"] == target_scheme for p in known_points):
                    continue

                pred_p50 = linear_fit_and_predict(x_bits, [p["ttft_p50_ms"] for p in known_points], target_bit)
                pred_p95 = linear_fit_and_predict(x_bits, [p["ttft_p95_ms"] for p in known_points], target_bit)
                pred_thru = linear_fit_and_predict(x_bits, [p["throughput_tok_s"] for p in known_points], target_bit)
                pred_qual = linear_fit_and_predict(x_bits, [p["task_score_delta"] for p in known_points], target_bit)

                eval_method = known_points[0]["eval_method"]
                eval_sample_size = known_points[0]["eval_sample_size"]
                prefix_caching = known_points[0]["prefix_caching"]

                all_source = []
                for p in known_points:
                    all_source.extend(p["source_runs"])

                known_points.append({
                    "inference_engine": engine,
                    "quant_algo": target_algo,
                    "quant_scheme": target_scheme,
                    "bit_width": target_bit,
                    "ttft_p50_ms": max(0.1, pred_p50),
                    "ttft_p95_ms": max(0.1, pred_p95),
                    "throughput_tok_s": max(0.1, pred_thru),
                    "task_score_delta": min(0.0, pred_qual),
                    "eval_method": eval_method,
                    "eval_sample_size": eval_sample_size,
                    "prefix_caching": prefix_caching,
                    "source_runs": all_source
                })

        points.extend(known_points)

    # Filter points by constraints (p95 latency budget and quality floor)
    valid_points = points
    if constraint.max_latency_ms is not None:
        valid_points = [p for p in valid_points if p["ttft_p95_ms"] <= constraint.max_latency_ms]
    valid_points = [p for p in valid_points if (100.0 + p["task_score_delta"]) >= constraint.min_quality_pct]

    if not valid_points:
        return None

    # Sort by balanced multi-axis score (not quality-uber-alles)
    valid_points.sort(key=score_candidate, reverse=True)

    best = valid_points[0]

    # Confidence logic
    confidence_label = "interpolated"
    if best["eval_sample_size"] < 50:
        confidence_label = "interpolated (low sample)"

    return Recommendation(
        inference_engine=best["inference_engine"],
        quant_algo=best["quant_algo"],
        quant_scheme=best["quant_scheme"],
        expected_ttft_p50_ms=best["ttft_p50_ms"],
        expected_ttft_p95_ms=best["ttft_p95_ms"],
        expected_throughput=best["throughput_tok_s"],
        quality_delta_pct=best["task_score_delta"],
        eval_method=best["eval_method"],
        eval_sample_size=best["eval_sample_size"],
        prefix_caching=best["prefix_caching"],
        confidence=confidence_label,
        source_rows=best["source_runs"]
    )


def recommend(constraint: Constraint, dataset: List[Dict[str, Any]]) -> Optional[Recommendation]:
    # 1. Filter by hardware (exact match only — no silent fuzzy fallback)
    hw_lower = constraint.hardware.strip().lower()
    candidates = [row for row in dataset if row["hardware"] == hw_lower]

    # Match on the closest available workload shape first
    candidates = filter_by_closest_workload(candidates, constraint.prompt_tokens, constraint.output_tokens)

    # 2. Filter by prefer_engine if specified
    if constraint.prefer_engine:
        candidates = [
            row for row in candidates
            if row["inference_engine"] == constraint.prefer_engine.strip().lower()
        ]

    # 3. Filter by model size (exact match)
    exact_candidates = [
        row for row in candidates
        if abs(row["model_size_b"] - constraint.model_size_b) < 1e-5
    ]

    # 4. Filter by max latency if specified (checked against p95)
    filtered_exact = exact_candidates
    if constraint.max_latency_ms is not None:
        filtered_exact = [
            row for row in filtered_exact
            if row["ttft_p95_ms"] <= constraint.max_latency_ms
        ]

    # 5. Filter by quality percentage
    filtered_exact = [
        row for row in filtered_exact
        if (100.0 + row["task_score_delta"]) >= constraint.min_quality_pct
    ]

    if filtered_exact:
        # Sort by balanced multi-axis score (not quality-uber-alles)
        filtered_exact.sort(key=score_candidate, reverse=True)
        best = filtered_exact[0]

        # Confidence logic
        confidence_label = "exact"
        if best["eval_sample_size"] < 50:
            confidence_label = "exact (low sample)"

        return Recommendation(
            inference_engine=best["inference_engine"],
            quant_algo=best["quant_algo"],
            quant_scheme=best["quant_scheme"],
            expected_ttft_p50_ms=best["ttft_p50_ms"],
            expected_ttft_p95_ms=best["ttft_p95_ms"],
            expected_throughput=best["throughput_tok_s"],
            quality_delta_pct=best["task_score_delta"],
            eval_method=best["eval_method"],
            eval_sample_size=best["eval_sample_size"],
            prefix_caching=best["prefix_caching"],
            confidence=confidence_label,
            source_rows=exact_candidates
        )

    return interpolate(constraint, dataset)
