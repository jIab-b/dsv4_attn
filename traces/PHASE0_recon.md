# Phase 0 — Recon

Date: 2026-05-20

## DeepSeek-V4 (the target)

- Released 2026-04-24, MIT, two SKUs:
  - **V4-Flash**: 285B total / ~13B active, FP4 MoE experts + FP8 attention. Fits TP=4 on a single GB200/B200 node.
  - **V4-Pro**: 1.6T / ~49B active, 1M context. TP=8 minimum.
- Architecture: hybrid **CSA** (Compressed Sparse Attention, small `m`, top-k DSA selection) + **HCA** (Heavily Compressed Attention, `m'=128`, dense over compressed entries), interleaved across layers. Both branches also carry a sliding-window attention (SWA) tail for local context. Per-head learnable sink σ_h folded into denominator. mHC (Manifold-Constrained Hyper-Connections) on residual stream.
- Paper: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" (2026-04-24).
- Day-0 sglang + vLLM support.

This grounds the sketch in `../sketch_hca.txt` and the `HcaParams` struct in `../hca/params.h`:
- `m'=128`, `W=128`, `H=128`, `c_out=512` ✅ matches paper.
- Hybrid rope (`rope_theta_compressed=160000`, `rope_theta_swa=10000`) ✅.
- Per-head `attn_sink[H]` ✅.
- FP8 e4m3 latent + e8m0 microscale block scales ✅ matches DSv4's "FP8 attention" wording.

## SGLang launch flags (from sglang cookbook)

V4-Flash low-latency, B200, TP=4:
```
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --chunked-prefill-size 4096 \
  --disable-flashinfer-autotune \
  --swa-full-tokens-ratio 0.1 \
  --host 0.0.0.0 --port 30000
```

For the trace I'll **disable EAGLE speculation** to keep the kernel mix clean (otherwise the trace gets dominated by draft-model kernels and the verify pass aliases what we want to measure).

## nsys profile pattern for sglang

Community-tested invocation (from sglang issues #2776, #7777, #8017):
```
nsys profile \
  -t cuda,nvtx,cudnn,cublas,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o /vol/traces/dsv4_decode \
  -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  python -m sglang.bench_one_batch_server [...]
```

Key gotchas:
- `--cuda-graph-trace=node` is required, otherwise cudagraph-launched kernels show up as a single fat node.
- `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` to capture sglang's unregistered NVTX ranges.
- Use `--capture-range=cudaProfilerApi` + sglang's `/start_profile` endpoint (which calls `cudaProfilerStart`) so we skip model-load time and only capture steady-state decode.

## Modal image

Use `lmsysorg/sglang:b200-cu129` (or pinned `v0.5.9-cu129-b200`). It ships with sglang, FlashInfer FP4 backends, NCCL, and CUDA 12.9. Add nsys from the CUDA toolkit (already in the image) or `pip install nvidia-nsight-systems` on top.

## Modal auth status

- modal CLI authenticated ✅
- `huggingface-secret` already exists in modal ✅
- HF cache will live on a modal Volume so we don't re-download 142 GB on every cold start.

## Decisions

- Model: **DeepSeek-V4-Flash** (cheaper, same arch as Pro, same kernels — Pro just has bigger experts).
- GPUs: **4x B200**, TP=4. Matches the cookbook's TP=4 recipe.
- Trace target: ~15s of steady-state decode at batch=8, prompt=512, decode_len=128. Trims to ~100–300 MB rep.
- No speculation in the traced run (cleaner kernel mix). I'll do a side-by-side check without speculation later if useful.
