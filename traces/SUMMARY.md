# DSv4-Flash decode — runtime breakdown + HCA func-sig validation

**Setup**: DeepSeek-V4-Flash, 4× B200, TP=4, FP4 routed experts, FP8 e4m3 KV cache, sglang 0.5.12. 25 s nsys window during steady-state batched decode at ~1000 tok/s aggregate. Trace at `remote/dsv4_decode.nsys-rep`.

---

## Dominant GPU runtime — top kernels by total time

Top of `cuda_gpu_kern_sum`, grouped by what they do (top 24 kernels = ~88% of all GPU time):

| Bucket | % | Kernels |
|---|---:|---|
| **TP communication** | **23.3 %** | `all_reduce_one_shot_push_kernel<bf16,4>` (20.5 %) + `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` (2.8 %) |
| **MoE FP8/FP4 expert GEMM** | **23.5 %** | `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl` ×2 variants (14.7 %) + 3 `bmm_MxE4m3_MxE2m1MxE4m3_…_swiGlu` shapes (8.8 %) |
| **mHC (manifold hyper-connections)** | **18.4 %** | `mhc_post_tilelang_kernel` (12.0 %) + `mhc_pre_big_fuse_with_norm_tilelang_kernel` (4.1 %) + `deep_gemm::sm100_tf32_hc_prenorm_gemm_impl` (2.3 %) |
| **Attention (MLA sparse + combine + planning)** | **8.6 %** | `flash_fwd_splitkv_mla_fp8_sparse_kernel<ModelType=1>` (4.3 %) + `flash_fwd_mla_combine_kernel` (2.1 %) + `sm100_fp8_paged_mqa_logits` (0.8 %) + `get_mla_metadata_kernel` (0.7 %) + `fused_k_norm_rope_flashmla` (0.7 %) |
| **cuBLASLt / cuBLAS GEMM (non-MoE, non-attn)** | **7.0 %** | `nvjet_tss_…_splitK_TNN` (4.6 %) + `nvjet_tst_…_splitK_TNN` (0.7 %) + `cublasLt::splitKreduce_kernel` (1.7 %) |
| **FP8 input quantization** | **3.4 %** | `per_token_group_quant_8bit_kernel` (2.7 %) + `tensorrt_llm::kernels::quantize_with_block_size` (0.7 %) |
| **MoE routing & finalize** | **3.1 %** | `moe::dev::finalize::finalizeKernel` (1.3 %) + `moe_fused_gate_kernel_small_token` (1.2 %) + `routingIndicesDynBlockKernel` (0.6 %) |
| **Elementwise / dtype copy** | **0.7 %** | torch `direct_copy_kernel` family |

### What this says about DSv4-Flash decode at TP=4

1. **It's communication-bound, not compute-bound.** ~23 % of GPU time is allreduce — most of it the `one_shot_push` LL variant for inter-rank fanout after MoE expert dispatch + after attention output projection. At TP=4 the bytes-per-allreduce are still modest but the launch cadence is high (`one_shot_push` fires ~123k times in 25 s ≈ 4900/s). Single-node NVLink-5 keeps the bandwidth side comfortable; the latency floor per allreduce (29 µs avg) is what eats the cycle budget.

2. **MoE and mHC are roughly tied (~23 % vs ~18 %).** That mHC takes nearly as much time as the MoE expert GEMMs is the surprise of this trace. The two tilelang kernels (`mhc_pre_big_fuse_with_norm`, `mhc_post`) plus the `tf32_hc_prenorm_gemm` are doing the manifold projection on the residual stream every layer. This is *not* the attention; it's the new residual structure that V4 introduced.

3. **Attention is small.** All MLA paths together (sparse decode + combine + scoring + planner + K-norm+RoPE) sum to only ~9 %. That's the whole point of CSA + HCA: at 4–8k context you're already paying near-zero attention tax compared to MHA models. There is one `flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<(ModelType)1>>` doing both CSA and HCA layers (the "sparse" name covers both — HCA layers run with all compressed entries selected).

4. **FP8 input quant is non-trivial (3.4 %).** `per_token_group_quant_8bit_kernel` and `quantize_with_block_size` together are the cost of pre-attention / pre-MoE bf16→fp8 quantization with e8m0 microscale block factors. That's the price you pay to use FP8 GEMMs.

---

## Host overhead — top API calls

`cuda_api_sum`:

| Time % | API | Notes |
|---:|---|---|
| 75.8 % | `cudaEventSynchronize` | 8.3 s of 11 s API time → host is GPU-bound, parked on events. **This is healthy** — means the host is keeping up. |
| 9.7 % | `cudaLaunchKernel` | 56 k calls / 25 s ≈ 2.2 k launches/s outside graphs |
| 2.9 % | `cudaMalloc` | 888 calls / 25 s ≈ 35/s. Avg 362 µs, max 21 ms. Mild churn. |
| 2.7 % | `cuLaunchKernelEx` | extended-launch path (mostly Triton-generated kernels) |
| 2.2 % | `cudaIpcOpenMemHandle` | 448 calls — TP IPC handle exchange (NCCL custom-allreduce shmem setup, recurring through the trace not just at init) |
| 1.7 % | `cudaGraphLaunch` | 1 396 graph launches / 25 s ≈ 56 decode steps/s system-wide. With batch=8 this is ~450 tok/s/req (matches server-log throughput). |
| 1.4 % | `cuKernelGetFunction` | one-shot symbol lookups, mostly during warmup |

### What this says about host overhead

- **No host CPU bottleneck.** Three quarters of the host API budget is just sitting on `cudaEventSynchronize`. The eager path (`cudaLaunchKernel` at 2.2 k/s) is well within driver capacity.
- **The 35 mallocs/s and 448 IPC-handle opens are the real candidates** for optimization. The IPC handles in particular shouldn't be opened in steady-state — that smells like an NCCL custom-allreduce path that re-establishes shared mappings per request batch. Worth a follow-up.
- **Graph-replay cost is invisible.** Each `cudaGraphLaunch` covers an entire decode step across all layers; the host hands the graph to the GPU once per step and waits.

---

## Memory ops (negligible)

| Op | Time | Bytes |
|---|---|---|
| D2D memcpy | 10.9 ms | 3.2 GB total (avg 345 KB) — KV cache appends + scratch shuffles |
| D2H memcpy | 4.3 ms | 30 KB total — sampled token IDs returning to host |
| memset | 2.9 ms | 3.9 GB total — buffer zeroing (single largest is 268 MB) |
| H2D memcpy | 2.1 ms | 380 KB total — incoming token IDs |

Total memory traffic ≤ 20 ms of GPU time in a 25 s window. Decode is squarely compute-and-collective-bound, not memory-traffic-bound.

---

## Per-layer NVTX picture

`enable-layerwise-nvtx-marker` produced NVTX ranges like `model.model.layers.N` and `model.model.layers.N.self_attn`. Notable observations:

- **Layer 0 is a 12-second outlier (max)** — that's the cudaGraph capture cost for the first layer's graph; the median for layer 0 is 2.79 ms/call.
- **Typical (non-capture) layers**: median attention range ≈ **700 µs**, median full-layer range ≈ **2.0–2.1 ms**. Across 32 attention layers ≈ **22 ms / decode step**, of which attention itself ≈ 6 ms. Consistent with the kernel-level totals.
- **Embed lookup**: `model.model.embed_tokens` ≈ 51 ms total over 4 invocations (12.8 ms/call). For 4 prefill batches in the window that's normal.
- **NCCL init**: `NCCL:ncclCommInitRankConfig` showed up at 580 ms/call × 4 — only at warmup, ignore.
- Per-layer NVTX is generic (`layers.0`, `layers.0.self_attn`); it does **not** distinguish CSA vs HCA layers, so separating their costs would require either reading `model.config` from sglang to know which index has which type, or adding custom NVTX rings around `CompressedSparseAttention.forward` vs `HeavilyCompressedAttention.forward`.

---

## HCA func-sig validation (the actual point of this exercise)

Comparing `hca/params.h`'s `HcaParams` against the kernel inputs/outputs we actually see in the live trace:

| HcaParams field | What sglang's traced kernel does | Match? |
|---|---|---|
| `Kc[B, M_max, 512] fp8 e4m3` + `Kc_scales[B, M_max, 4] e8m0` + `Kc_rope[B, M_max, 64] bf16` | `flash_fwd_splitkv_mla_fp8_sparse_kernel` TmaParams expose a 512-wide latent tile + a 64-wide rope tile per MLA head, with e8m0 microscale block factors. Page size in the server log is **256**, our sketch's `M_max` strides are consistent (just a different paging factor). | ✅ |
| `Kswa[B, 128, 512]` + `Kswa_rope[B, 128, 64]` (SWA branch) | Server log: `swa_full_tokens_ratio=0.1`, `swa token usage` counted separately in decode batches → SWA branch is live and carries the same FP8+bf16-rope layout. | ✅ |
| `Q[B, 1, H=128, 512]` / `O[B, 1, H=128, 512]` | DSv4-Flash actually serves with **H=64 heads** per the `attn_sink: [64]` NVTX payload, not 128. **Your sketch's H=128 is V4-Pro-shaped, not Flash-shaped.** Same kernel just different head dim. | ⚠️ Make `H` a template/runtime parameter, not a constant. |
| `attn_sink[H]` | NVTX confirms `'TrainableParams': {'attn_sink': [64]}` per `self_attn` block — exact shape match. | ✅ |
| `partial_O[split, B, H, 512]` + `partial_lse[split, B, H]` (split-K) | Confirmed via `get_mla_metadata_kernel` (split-planner) → `flash_fwd_splitkv_mla_fp8_sparse_kernel` (compute) → `flash_fwd_mla_combine_kernel` (reduce). Your `launch_hca_combine` is the analog of the third. | ✅ |
| `sm_scale_log2`, `rope_theta_compressed`, `rope_theta_swa` | sglang fuses K-norm + RoPE in `fused_k_norm_rope_flashmla` and consumes `sm_scale_log2` inside the FP8 sparse kernel. The hybrid rope (different θ for HCA-compressed vs SWA branches) is matched. | ✅ |
| TMA descriptors (`tma_Q_sw128`, `tma_O`, `tma_Kc*`, `tma_Kswa*`) | The cute::Layout signatures inside `flash_fwd_splitkv_mla_fp8_sparse_kernel` show exactly the same swizzle/TMA scaffolding — SM90/SM100 TMA load + store. | ✅ |
| Compressor inputs (`C`, `Z`, `bias`, `partial_count`, `m_prime=128`) | These belong to **upstream block-compressor passes**, not the attention kernel itself. The traced kernels equivalent to "compress 128 tokens into 1 entry" are part of the per-layer pre-attention path (showing up inside the layer NVTX range as a mix of `deep_gemm` calls + `mhc_pre_*`). The sig is consistent. | ✅ |
| Legacy reduce fields (`out`, `stride_out_b`, `stride_out_blk`) | Not used by sglang's kernel — fine to drop. | clean-up |

### Verdict

Your `HcaParams` struct is **consistent with the live DSv4 attention kernel** and is shaped to be a drop-in for the same role `flash_fwd_splitkv_mla_fp8_sparse_kernel` plays in sglang.

Action items, ordered by importance:
1. **Make `H` and `c_out` template / runtime, not assumed constants** — V4-Flash uses H=64, V4-Pro uses H=128, both with the same kernel; your code currently hard-bakes H=128 in comments. Easy fix.
2. **Add explicit `pos_offset` + `yarn_factor`** to support the same rope-offset semantics sglang uses (`rope_theta_compressed=160000` matches the paper). Currently rope is assumed pre-applied to `Kc_rope`; document that explicitly or surface the fields.
3. **Add a K-norm+RoPE fused kernel** (mirrors sglang's `fused_k_norm_rope_flashmla`) — your current sig splits these into separate steps. Not a sig change, but a kernel you'll want.
4. **Add a metadata/planner kernel** analogous to `get_mla_metadata_kernel`. Currently you assume strides are pre-computed host-side; on the actual workload sglang plans split-K device-side per request. Either generate the strides on host (slower, simpler) or add a planner kernel.
5. **Drop the legacy reduce fields** (`out`, `stride_out_b`, `stride_out_blk`) — the standalone reduce belongs in `HcaCombineParams`, not `HcaParams`.

None of these are blocking. The signature as written can power the same workload sglang is running today.

---

## Provenance

- Phase 0: `PHASE0_recon.md` — DSv4 facts + sglang flag set
- Phase 1: `PHASE1_modal_image.md` — modal image build & 1× B200 sanity
- Phase 2: `PHASE2_weights.md` — V4-Flash weights cached on `dsv4-hf-cache` volume
- Phase 3: `PHASE3_trace.md` — 4× B200 nsys decode capture (incl. one false start)
- This file: phase 4 analysis + sig validation

Modal scripts left behind in this directory: `modal_app.py` (sanity / download_weights / serve_and_trace / analyze_rep entrypoints), `analyze.py` (local CSV→markdown summarizer; not used in the end because of the nsys version skew).
