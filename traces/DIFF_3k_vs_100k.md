# Decode-trace diff: ~3 k vs ~31 k context

Two traces under identical sglang/nsys flags, only the prompt length and bench batch differ:

| | 3 k run (`dsv4_decode.nsys-rep`) | 100 k run (`dsv4_decode_100k.nsys-rep`) |
|---|---|---|
| Prompt length per req | ~3 k tokens (`#full token: 2560` for batch=8) | ~31 k tokens (`#full token: 62 464` for batch=2) |
| Bench batch | 8 | 2 |
| Decode rate (server log) | 1017–1034 tok/s aggregate | 284 tok/s aggregate |
| Per-decode-step latency | ~ 0.97 ms | ~ 3.52 ms (**× 3.6 slower**) |
| Capture window | 25 s | 25 s |
| Total kernel-time captured (sum across ranks/streams) | 17.65 s | 15.50 s |
| Rep size | 120.7 MB | 99.9 MB |

The 100 k run terminated cleanly through the nsys window, but the GPUs raised XID 13/38/109/154 (`CTX SWITCH TIMEOUT` → `GPU Reset Required`) **after** the capture window ended — captured data is valid, but DSv4-Flash on sglang 0.5.12 + B200 isn't stable past ~30 k tokens in this configuration. Worth filing upstream.

---

## Per-role kernel diff (>= 0.4 % in either run)

Sorted by max share. `Δ pp` is long minus short, `×` is the ratio of absolute kernel-time (long / short).

```
role                              short %   long %    Δ pp     ×     short ms    long ms   short insts    long insts
-------------------------------------------------------------------------------------------------------------------
TP_ALLREDUCE_CUSTOM                20.52%   13.60%   -6.91   0.58     3621.5     2108.5      122,884        57,768
TP_ALLREDUCE_NCCL                   2.79%   16.13%  +13.34   5.07      493.0     2501.2          696         5,220
MOE_GEMM_FP8FP4                    14.94%   10.23%   -4.70   0.60     2636.7     1586.4      396,308       201,996
mHC_POST                           11.99%   10.78%   -1.21   0.79     2117.3     1671.9      122,156        62,264
ATTN_SHARED_(MLA over compr cache) 4.33%    10.47%   +6.14   2.12      764.6     1622.7       61,080        31,132
CUBLASLT_GEMM                       5.81%    4.13%   -1.68   0.62     1024.9      640.4      180,388        91,948
MOE_BMM_SWIGLU                      5.58%    4.90%   -0.68   0.77      984.5      759.7       61,080        31,132
mHC_PRE                             4.10%    3.22%   -0.88   0.69      723.7      499.5      122,160        62,264
MOE_BMM                             3.83%    3.28%   -0.55   0.75      676.9      509.2       61,078        31,132
FP8_QUANT_PER_TOKEN                 3.24%    2.54%   -0.70   0.69      571.2      393.5      396,308       201,996
ATTN_CSA_SCORE (paged_mqa_logits)   0.75%    3.15%   +2.40   3.68      132.5      488.0       29,828        15,204
ELEMENTWISE                         3.09%    2.24%   -0.86   0.64      546.3      347.1      309,488        72,332
mHC_PRENORM_GEMM                    2.41%    2.24%   -0.17   0.82      425.3      347.9      122,160        62,264
ATTN_SPLITK_COMBINE                 2.06%    1.35%   -0.71   0.58      363.6      209.6       61,080        31,132
CUBLASLT_SPLITK_REDUCE              2.05%    1.05%   -1.00   0.45      361.5      162.7      178,144        83,896
MOE_ROUTING_IDX                     1.47%    0.76%   -0.71   0.45      259.8      117.4      121,816        59,684
MOE_FINALIZE                        1.36%    1.40%   +0.05   0.91      239.4      217.3       61,076        31,132
MOE_GATING                          1.24%    0.80%   -0.44   0.57      218.1      124.0       56,808        28,960
ATTN_MLA_PLANNER                    0.69%    0.73%   +0.04   0.93      121.6      112.6        4,272         2,172
ATTN_CSA_QINDEXER                   0.35%    0.72%   +0.37   1.80       62.1      112.0       29,828        15,204
ATTN_KNORM_ROPE                     0.69%    0.50%   -0.19   0.63      122.0       77.4       61,080        31,132
ATTN_QNORM_ROPE                     0.58%    0.65%   +0.07   0.98      102.4      100.6       61,080        31,132
ATTN_ROPE_DSV3                      0.52%    0.56%   +0.04   0.95       91.6       86.7       61,080        31,132
ATTN_HCA_DECODE (flash_c4_decode)   0.58%    0.30%   -0.28   0.45      102.3       45.9       58,312        27,048
RMSNORM                             0.66%    0.40%   -0.25   0.54      115.8       62.6       62,500        31,856
FP8_QUANT_BLOCK                     0.70%    0.61%   -0.09   0.76      124.0       94.9       61,080        31,132
```

---

## The interesting movers

### 1) The MLA-over-compressed-cache kernel is *the* long-context cost

`flash_fwd_splitkv_mla_fp8_sparse_kernel<ModelType=1>` jumped from **4.33 % → 10.47 %** (× 2.12 in share). But share is a poor measure because the 100 k run also has fewer total decode steps. Normalising by per-decode-step:

- short: 764.6 ms / ~1.87 k decode steps ≈ **0.41 ms / step**
- long:  1622.7 ms / ~0.97 k decode steps ≈ **1.67 ms / step** (**× 4.1**)

That ~4× growth ≈ matches the compressed-entry-count ratio (31 k / 128 vs 3 k / 128 ≈ 10×; CSA's smaller `m=4` ratio is ≈ 31 k / 4 vs 3 k / 4 ≈ 10×; but the kernel is split-K and TMA-streamed so it scales better than 1:1 with entries — the 4× we see is consistent with the FlashAttention-style sublinear scaling).

**This is where DSv4's compressed attention actually pays the long-context cost** — both CSA *and* HCA layers route through this kernel.

### 2) CSA scoring (`paged_mqa_logits`) grew almost in lockstep

`deep_gemm::sm100_fp8_paged_mqa_logits` went **0.75 % → 3.15 %** (× 3.68 in share; × 3.68 per step too since instance counts halved together). This is the kernel that computes the score matrix used to pick the top-k compressed entries for CSA layers. It's purely a function of compressed-entry count, so a × 10 in entries → × 3.7 in time (sublinear, classic TMA-fed GEMM).

### 3) `flash_c4_decode` did **not** grow — and that overturns my earlier guess

I had labeled `flash_c4_decode<(long)128>` / `<(long)512>` as "HCA decode" in the previous summary. The per-instance time tells a different story:

- short: 102.3 ms / 58 312 inst = **1.75 µs/inst**
- long: 45.9 ms / 27 048 inst = **1.70 µs/inst**

**Per-instance cost is essentially unchanged.** A kernel that does dense attention over compressed entries would scale with context, even if logarithmically. The correct interpretation, given the data:

> `flash_c4_decode` is the **per-decode-step Compress-4 update kernel** — it compresses the *new* K/V token into the running compressed cache (and/or maintains the rope-side). It is *not* the kernel that reads the compressed cache during attention.

The actual reader for both CSA's selected-entries-attention and HCA's dense-over-compressed-entries is `flash_fwd_splitkv_mla_fp8_sparse_kernel`. "sparse" here describes the *memory-access pattern* (indexed via paged-MQA tables), not whether the op is selective. The same kernel handles both layer types via different params.

This re-attribution is consistent with the kernel signatures' param structs: `SparseAttnDecodeParams` is much larger than `Compress4DecodeParams`, with token-range tables that would only be needed for the actual attention read.

### 4) TP communication routing flipped, not grew

- `TP_ALLREDUCE_CUSTOM` (the `one_shot_push` bf16 LL fanout): 20.5 % → 13.6 %
- `TP_ALLREDUCE_NCCL` (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`): 2.8 % → 16.1 %

Total TP comm: **23.3 % → 29.7 %** (mild grow, attention got bigger so relative share of comm also drifted). The interesting bit is the *routing flip*: at batch=2 sglang's threshold for the custom one-shot path apparently isn't met, so it falls back to NCCL's RING_LL kernel. The 696 → 5 220 instance count jump (× 7.5) on the NCCL kernel is a strong signal — there's a tunable here.

### 5) MoE and mHC shrunk in absolute time

MoE FP8/FP4 GEMM: 2 637 ms → 1 586 ms (× 0.60). MoE BMM-swiGlu: 985 → 760 ms (× 0.77). mHC_POST: 2 117 → 1 672 ms (× 0.79). All scale with `batch × tokens / step`, which dropped from 8 to 2. Per-token MoE cost is roughly flat; total time shrinks proportionally to batch.

### 6) Per-decode-step rebalance summary

Aggregating *per decode step* (kernel-ms / number of decode steps in the window, all 4 ranks summed):

| Bucket | short (ms/step) | long (ms/step) | × |
|---|---:|---:|---:|
| ATTN_SHARED (compressed-cache read) | 0.41 | 1.67 | **× 4.1** |
| ATTN_CSA_SCORE (top-k scoring) | 0.071 | 0.502 | **× 7.1** |
| ATTN_SPLITK_COMBINE | 0.194 | 0.216 | × 1.1 |
| ATTN_CSA_QINDEXER | 0.033 | 0.115 | × 3.5 |
| ATTN_HCA_DECODE (c4 incremental update) | 0.055 | 0.047 | × 0.86 |
| MoE GEMM+BMM+routing (per step) | ~2.3 | ~2.7 | × 1.17 |
| mHC (per step) | ~1.8 | ~2.6 | × 1.46 |
| TP allreduce total (per step) | ~2.2 | ~4.7 | × 2.15 |

Per step the **attention cost grew from ~0.7 ms to ~2.5 ms** (×3.5), and **TP allreduce roughly doubled** because the larger attention outputs need more allreduce traffic. Together those two account for nearly all the decode-rate slowdown (~3.6× per step → ~3.5× lower tok/s).

---

## Implications for the HCA func sig

The instance-count math (`flash_c4_decode` fires twice per HCA layer per decode step regardless of context length) reinforces a piece of the sketch — the HCA layer maintains a **per-step incremental compressor**. Two related shape constraints for `HcaParams`:

1. There's a **separate "compress" kernel** that consumes the in-flight uncompressed block (`C`, `Z`, `bias`, `partial_count`) and emits one compressed entry every `m'=128` tokens. That's `flash_c4_decode<(long)128, ..., Compress4DecodeParams>` in the trace. Your sketch's `C`/`Z`/`bias` fields are correct; the planner state (`M_cur`, `partial_count`) is also correct.

2. The **attention read** (your `Q`/`Kc`/`Kc_rope`/`Kswa`/etc. fields) flows through a *different* kernel — the FlashMLA-flavored sparse-indexed kernel — that is **not the same** as the compressor. In your `params.h` you've correctly kept these as separate concerns (the attention struct vs the compressor inputs at the top), so this matches sglang's layout.

What you still want to add for parity:
- A planner-kernel analog (sglang's `get_mla_metadata_kernel` + `smxx_paged_mqa_logits_metadata`) that produces the per-request stride tables / split-K plans, instead of computing them host-side.
- An explicit CSA-vs-HCA `mode` field (or two distinct entry points) since DSv4 interleaves layer types and the same struct shouldn't be ambiguous about which branch it's serving.

---

## Files dropped

- `dominant_kernels.txt` — top kernels w/ full sigs (3 k run + addendum w/ all 13 attention kernels)
- `DIFF_3k_vs_100k.md` — this file
- `diff_3k_vs_100k.txt` — raw per-bucket numbers, machine-readable
- `remote/dsv4_decode_100k.nsys-rep` — the long-context rep (open in nsys 2026.2+ GUI)
- `remote/dsv4_decode_100k_stats_*.csv` — per-report CSVs
- `remote/server_100k.log`, `remote/bench_100k.log` — server + bench logs
- `modal_app.py` now has `trace_100k` and `analyze_100k` entrypoints
