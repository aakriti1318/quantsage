![alt text](https://raw.githubusercontent.com/aakriti1318/sage-quant/main/image.png)

<p align="center">
  <a href="https://pypi.org/project/sage-quant/"><img src="https://img.shields.io/pypi/v/sage-quant?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://github.com/aakriti1318/sage-quant"><img src="https://img.shields.io/github/stars/aakriti1318/sage-quant?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/aakriti1318/sage-quant/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License MIT"></a>
</p>

**SageQuant** is a data-driven CLI tool that helps you choose the optimal serving stack. Given your model size, target hardware, workload shape, and tail latency/quality budget, it recommends the best **inference engine**, **quantization algorithm**, and **bit-width scheme**.

It provides choices backed by real benchmark data, not guesswork.

---

## Why SageQuant?

Choosing how to serve an LLM today is usually an ad-hoc decision: defaulting to vLLM, trying AWQ `w4a16` because of a blog post, and hoping for the best. 

SageQuant replaces this guesswork with a structured lookup across three distinct decisions:

1. **Inference Engine**: vLLM vs. SGLang vs. TensorRT-LLM vs. MLX
2. **Quantization Algorithm**: GPTQ vs. AWQ vs. FP8 vs. SmoothQuant vs. MLX native
3. **Bit-width Scheme**: FP16 vs. 8-bit (W8A8) vs. 4-bit (W4A16)

---

## Tradeoff Visualization

SageQuant maps tradeoffs across engines and algorithms for the same model and hardware:

| Engine | Quant Algo | Scheme | TTFT (p50 / p95) | Throughput | Quality vs. FP16 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vLLM** | none | fp16 | 85 / 140ms | 45.0 tok/s | baseline |
| **vLLM** | GPTQ | w4a16 | 145 / 210ms | 38.2 tok/s | -1.1% (mmlu-5shot, n=200) |
| **vLLM** | AWQ | w4a16 | 140 / 205ms | 39.5 tok/s | -0.8% (mmlu-5shot, n=200) |
| **SGLang** | AWQ | w4a16 | 98 / 165ms | 52.0 tok/s | -1.4% (mmlu-5shot, n=200) |

![alt text](https://raw.githubusercontent.com/aakriti1318/sage-quant/main/workflow.png)

---

## Quick Start

### 1. Install
```bash
pip install -e .
# Advanced features (Streamlit dashboard & scikit-learn regression interpolation):
pip install -e ".[advanced]"
```

### 2. Get a Recommendation
Find the best stack based on your budget:
```bash
sage-quant recommend --model-size 7b --hardware a100-40gb --max-latency 200ms --min-quality 98
```
```
Recommended: SGLANG + AWQ (w4a16), prefix caching on
Workload: 128 in / 128 out tokens
Expected: 60ms TTFT (p50) · 105ms TTFT (p95) · 75.0 tok/s · -1.4% quality (mmlu-5shot, n=200)
Confidence: exact (3 matching benchmark runs)

Note (Engine): SGLang features RadixAttention and is highly efficient for high-cache-reuse workloads.
Note (Algo): AWQ generally preserves quality slightly better than GPTQ at 4-bit.
```

### 3. Generate Serving Config
Get a copy-pasteable server launch command:
```bash
sage-quant serve-config --model-size 7b --hardware a100-40gb --model meta-llama/Meta-Llama-3-8B-Instruct --prefer-engine sglang
```

### 4. Contribute Benchmarks
If you have run benchmarks locally, you can append your run log and share it with the community. You can refer to [my_run.json](my_run.json) for the expected input JSON format:
```bash
sage-quant contribute --run-log my_run.json
```
This command automatically validates your data, appends it locally, and generates a zero-friction, pre-filled link to open a GitHub Issue with your contribution:

![GitHub Issue Pre-filled Contribution](https://raw.githubusercontent.com/aakriti1318/sage-quant/main/contribute_issue.png)

---

## Command Reference

### `sage-quant recommend`
Find the optimal engine, algorithm, and scheme based on latency/quality constraints.
```bash
sage-quant recommend --model-size 7b --hardware a100-40gb --max-latency 200ms --min-quality 98 --prompt-tokens 128 --output-tokens 128
```

### `sage-quant serve-config`
Generate runnable vLLM/SGLang/MLX launch configuration scripts or YAML files.
```bash
sage-quant serve-config --model-size 7b --hardware a100-40gb --model meta-llama/Meta-Llama-3-8B-Instruct
```

### `sage-quant list-hardware`
List all unique hardware configurations in the dataset.
```bash
sage-quant list-hardware
```

### `sage-quant list-engines`
List all unique inference engines in the dataset.
```bash
sage-quant list-engines
```

### `sage-quant list-quant-algos`
List all unique quantization algorithms in the dataset.
```bash
sage-quant list-quant-algos
```

### `sage-quant contribute`
Append custom benchmark runs to the local dataset and get zero-friction sharing instructions.
```bash
sage-quant contribute --run-log my_run.json
```

---

## Documentation

For a comprehensive guide, detailed command usage, and real-world example scenarios, see the [User Guide](docs.md).

---

## License

MIT