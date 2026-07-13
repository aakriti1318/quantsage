![alt text](https://raw.githubusercontent.com/aakriti1318/quantsage/main/image.png)

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

![alt text](https://raw.githubusercontent.com/aakriti1318/quantsage/main/workflow.png)

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

---

## Command Reference

- `sage-quant recommend` — Find optimal engine, algorithm, and scheme. Supports `--prompt-tokens`, `--output-tokens`, and `--prefer-engine`.
- `sage-quant serve-config` — Generate vLLM/SGLang/MLX launch configuration scripts or YAML files.
- `sage-quant list-hardware` / `list-engines` / `list-quant-algos` — Check what combinations are currently covered in the dataset.
- `sage-quant contribute` — Append custom benchmark runs (`JSON`/`CSV`) or run automated live tests using `[benchmark]` extra.

---

## Documentation

For a comprehensive guide, detailed command usage, and real-world example scenarios, see the [User Guide](docs.md).

---

## License

MIT