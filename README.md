# tinyinfer

A from-scratch GPT-2 inference engine in C++17, with no ML library dependencies.

> Work in progress. Benchmarks will go here once there is something to measure.

## Quick start

```bash
# 1. Python env for exporting weights (one time)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/export_gpt2.py        # writes models/gpt2.bin + models/gpt2_ref.bin

# 2. Build and test
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tinyinfer models/gpt2.bin
```

## Layout

| Path | What |
|---|---|
| `scripts/export_gpt2.py` | Exports Hugging Face GPT-2 to a flat binary, plus PyTorch reference logits |
| `include/tinyinfer/model.h`, `src/model.cpp` | Config, weight structs, and the loader |
| `include/tinyinfer/ops.h`, `src/ops.cpp` | layernorm, linear, GELU, softmax, and the forward pass |
| `tests/` | GoogleTest: loader checks, op unit tests, full forward pass vs. PyTorch |

## Weight format

256-byte header (magic `TINF`, version, then layer/head/embedding/context/vocab sizes),
followed by float32 tensors. Linear weights are stored `[out, in]`, so each output is a dot
product over one contiguous row. See the docstring in `scripts/export_gpt2.py` for the full order.

## Roadmap

- [x] Weight export and loader
- [ ] Naive forward pass matching PyTorch to 1e-3
- [ ] Generation loop (greedy, temperature, top-k) and baseline benchmark
- [ ] KV cache
- [ ] Fast matmul: tiling, multithreading, NEON SIMD
- [ ] int8 weight quantization, with perplexity before and after
- [ ] CI
