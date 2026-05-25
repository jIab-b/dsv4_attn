# Phase 1 — Modal image + sanity check

Status: **DONE**

## What was built

- Image: `lmsysorg/sglang:v0.5.12-cu129` (DSv4 Day-0 era, Blackwell/sm100 native)
- Added: `requests`, bumped `typing_extensions>=4.14.0` (fixes a `Sentinel` ImportError that bit on cold start)
- Env: `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` so sglang's unregistered NVTX ranges show up, `HF_HUB_CACHE=/cache/hf` pointing into a persistent Volume
- Volumes: `dsv4-hf-cache` (HF weights), `dsv4-traces` (nsys output)
- Secret: `huggingface-secret` (already in modal)

## First gotcha

`lmsysorg/sglang:v0.5.9-cu129-b200` didn't exist on Docker Hub. The b200-suffixed tags were retired sometime before v0.5.12 — the plain `cu129` tag is Blackwell-capable. Switched to `v0.5.12-cu129`.

## Sanity output (1x B200)

```
NVIDIA B200, 183 GB HBM, driver 580.95.05, CUDA 13.0 driver
nsys version 2026.2.1.210
torch 2.11.0+cu129, sm100
sglang 0.5.12
nccl 2.28.9
```

All boxes ticked: B200 surfaces as `sm100`, torch is cu129, NCCL 2.28 is the multi-node-comm-stable build, nsys 2026.2 supports `--cuda-graph-trace=node` and B200 SM counters.

Cost so far: ~$0.10.

## Next

Phase 2: download V4-Flash weights (~140 GB) into the HF cache volume. One-shot, then cached for all subsequent runs.
