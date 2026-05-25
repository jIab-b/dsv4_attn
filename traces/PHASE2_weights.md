# Phase 2 — Weight download

Status: **DONE**

## Outcome

Pulled `deepseek-ai/DeepSeek-V4-Flash` into the `dsv4-hf-cache` volume on a single B200.

- 74 blobs total in `models--deepseek-ai--DeepSeek-V4-Flash/blobs/`
- Snapshot dir is symlinks → blobs, so the local `du -sh` showed only 316K of symlinks (a red herring); the actual blobs total to ~150 GB.
- HF XET high-perf downloaded the 74 shards in ~90s — well under the function's 60-min timeout.

## Gotcha

HF warned "sending unauthenticated requests" despite the `huggingface-secret` mount. Download still completed because V4-Flash is MIT/public. For higher-rate-limit safety on re-downloads I should verify the secret stores its token under the `HF_TOKEN` key (rather than `HUGGINGFACE_TOKEN` etc.). Not blocking.

Cost so far (cumulative): ~$3 (mostly idle time during snapshot fanout).

## Next

Phase 3: launch sglang server on 4x B200 with TP=4, warm up, attach nsys for a 20s decode capture.
