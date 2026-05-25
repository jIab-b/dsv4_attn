"""Modal app: serve DeepSeek-V4-Flash on 4x B200 via sglang and capture an nsys decode trace.

Entrypoints:
    modal run modal_app.py::sanity                # cheap, ~1 min, env check
    modal run modal_app.py::download_weights      # one-shot weight pull into Volume
    modal run modal_app.py::serve_and_trace       # the real run

Outputs land in the `dsv4-traces` Volume as `dsv4_decode.nsys-rep` (and the bench JSON
next to it). Pull with `modal volume get dsv4-traces /traces/* ./` after the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

MINUTES = 60

# ---------- image ----------
# lmsysorg/sglang B200 image with CUDA 12.9, ships sglang + flashinfer + nccl + nsys.
sglang_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:v0.5.12-cu129",
    )
    .entrypoint([])
    .run_commands(
        # ensure nsys is on PATH (it ships in the CUDA toolkit base layer)
        "ls /opt/nvidia/nsight-systems/*/bin/nsys || which nsys || true",
        "nsys --version || true",
    )
    .uv_pip_install("requests==2.32.3", "typing_extensions>=4.14.0")
    .env(
        {
            "HF_HUB_CACHE": "/cache/hf",
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Capture sglang's unregistered NVTX ranges
            "NSYS_NVTX_PROFILER_REGISTER_ONLY": "0",
            # Quieter, more profile-friendly
            "TOKENIZERS_PARALLELISM": "false",
            "SGLANG_DISABLE_TQDM": "1",
        }
    )
)

# ---------- volumes ----------
hf_vol = modal.Volume.from_name("dsv4-hf-cache", create_if_missing=True)
trace_vol = modal.Volume.from_name("dsv4-traces", create_if_missing=True)

HF_CACHE_PATH = "/cache/hf"
TRACE_PATH = "/traces"

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"

app = modal.App("dsv4-nsys")


# ---------- helpers ----------
def _sglang_server_cmd(port: int = 30000) -> list[str]:
    """Low-latency-ish flags but NO speculation (cleaner kernel mix in the trace).

    Profile-relevant flags:
      --enable-profile-cuda-graph        emit kernels inside cudagraphs so nsys can see them
      --enable-layerwise-nvtx-marker     per-layer NVTX ranges so NVTX summary isn't empty
    """
    return [
        "python",
        "-m",
        "sglang.launch_server",
        "--trust-remote-code",
        "--model-path",
        MODEL_ID,
        "--tp",
        "4",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--chunked-prefill-size",
        "4096",
        "--disable-flashinfer-autotune",
        "--swa-full-tokens-ratio",
        "0.1",
        "--mem-fraction-static",
        "0.90",
        "--enable-profile-cuda-graph",
        "--enable-layerwise-nvtx-marker",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _wait_for_server(port: int, timeout_s: int) -> None:
    import requests

    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2)
    raise RuntimeError(f"sglang server didn't come up in {timeout_s}s (last err: {last_err})")


def _one_completion(port: int, prompt: str, max_new: int) -> dict:
    """Raw /generate (no chat template). Mostly for warmup load."""
    import requests

    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0.7, "max_new_tokens": max_new},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def _one_chat(port: int, user_msg: str, max_new: int) -> dict:
    """OpenAI-compatible chat endpoint — auto-applies DSv4 chat template."""
    import requests

    r = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": 0.0,
            "max_tokens": max_new,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


# ---------- entrypoints ----------
@app.function(
    image=sglang_image,
    gpu="B200:1",
    timeout=10 * MINUTES,
    volumes={HF_CACHE_PATH: hf_vol, TRACE_PATH: trace_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def sanity():
    """Cheap GPU / env check. Single B200, ~1 min."""
    print("=== nvidia-smi ===")
    subprocess.run(["nvidia-smi"], check=False)
    print("=== nsys ===")
    subprocess.run(["nsys", "--version"], check=False)
    print("=== python / sglang / torch ===")
    import torch  # noqa: WPS433

    print("torch", torch.__version__, "cuda", torch.version.cuda, "ndev", torch.cuda.device_count())
    try:
        import sglang  # noqa: WPS433

        print("sglang", sglang.__version__)
    except Exception as e:
        print("sglang import failed:", e)
    print("=== CUDA props ===")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  dev{i}: {p.name} sm{p.major}{p.minor} mem={p.total_memory/1e9:.1f}GB")
    print("=== nccl ===")
    try:
        print("nccl", torch.cuda.nccl.version())
    except Exception as e:
        print("nccl probe failed:", e)
    print("=== HF cache mount ===")
    subprocess.run(["ls", "-la", HF_CACHE_PATH], check=False)
    return "ok"


@app.function(
    image=sglang_image,
    gpu="B200:1",
    timeout=60 * MINUTES,
    volumes={HF_CACHE_PATH: hf_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_weights():
    """Pull V4-Flash weights into the HF cache volume. ~140 GB, single shot."""
    from huggingface_hub import snapshot_download

    print(f"snapshot_download {MODEL_ID} -> {HF_CACHE_PATH}")
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=HF_CACHE_PATH,
        max_workers=16,
    )
    print("done:", path)
    subprocess.run(["du", "-sh", path], check=False)
    hf_vol.commit()
    return path


@app.function(
    image=sglang_image,
    gpu="B200:4",
    timeout=75 * MINUTES,
    volumes={HF_CACHE_PATH: hf_vol, TRACE_PATH: trace_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def serve_and_trace(
    trace_duration_s: int = 25,
    delay_s: int = 540,
    bench_batch: int = 8,
    bench_input_len: int = 512,
    bench_output_len: int = 256,
    bench_prompt_token_target: int = 0,
    output_tag: str = "",
    dry_run: bool = False,
):
    """Launch sglang server *under* nsys with `--delay` skipping load+warmup, capture ~trace_duration_s.

    Why this shape:
        - nsys 2026.2 doesn't support attaching to a running pid (`--pid=` is rejected).
        - So we wrap the server from the start. `--delay=N` defers capture by N seconds (covers
          weight load + cudagraph capture + warmup), then `--duration=M` captures M seconds.
        - A bench client thread keeps the GPU busy through the whole capture window.

    Steps:
        1. Spawn `nsys profile --delay --duration -- python -m sglang.launch_server ...`.
        2. Sidecar thread waits for /health, fires chat warmup, then issues continuous decode load.
        3. nsys auto-stops; child server is SIGINT'd and exits.
    """
    import threading

    port = 30000
    rep_dir = Path(TRACE_PATH)
    rep_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_tag}" if output_tag else ""
    rep_path = rep_dir / f"dsv4_decode{suffix}"
    log_path = rep_dir / f"server{suffix}.log"
    bench_log = rep_dir / f"bench{suffix}.log"

    server_cmd = _sglang_server_cmd(port=port)
    nsys_cmd = [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx,cudnn,cublas,osrt",
        "--cuda-graph-trace=node",
        "--force-overwrite=true",
        f"--delay={delay_s}",
        f"--duration={trace_duration_s}",
        "--sample=none",
        "--cpuctxsw=none",
        "--output",
        str(rep_path),
        "--",
    ] + server_cmd
    print("wrapped cmd:", " ".join(nsys_cmd))
    if dry_run:
        print("dry_run=True, exiting before server launch")
        return {"dry_run": True}

    log_f = open(log_path, "w")
    server = subprocess.Popen(
        nsys_cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    print("nsys+server pid:", server.pid)

    stop_flag = threading.Event()
    bench_state: dict = {"ready_at": None, "warmup_done_at": None, "first_bench_at": None}

    def loader_loop():
        """Wait for /health, then drive warmup + continuous bench load."""
        import requests

        try:
            _wait_for_server(port, timeout_s=18 * MINUTES)
        except Exception as e:
            print("loader: server never became healthy:", e)
            return
        bench_state["ready_at"] = time.time()
        print("loader: server ready")

        # chat-style warmup (DSv4 is chat-tuned; raw /generate degenerates without a template)
        try:
            for i in range(3):
                out = _one_chat(port, "Write one short sentence about an octopus.", max_new=48)
                txt = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
                print(f"loader: warmup chat {i} -> {txt!r}")
            sanity = _one_chat(port, "What is the capital of France? Answer with one word.", max_new=8)
            sanity_text = (sanity.get("choices") or [{}])[0].get("message", {}).get("content", "")
            print(f"loader: sanity chat -> {sanity_text!r}")
        except Exception as e:
            print("loader: warmup error:", e)
        bench_state["warmup_done_at"] = time.time()

        # continuous decode load (raw /generate is fine here — we just want kernels firing)
        if bench_prompt_token_target > 0:
            # Long-context mode. ~4 chars/token for English; build a coherent-ish long prompt.
            # Repeat a Wikipedia-flavored paragraph until we hit the target token count.
            base = (
                "The hybrid attention mechanism in modern large language models compresses "
                "the key-value cache across the context window to enable efficient decoding. "
                "Researchers have proposed several variants, including compressed sparse attention "
                "and heavily compressed attention, which interleave across transformer layers. "
                "Each variant trades off between recall and compute, and the optimal mix depends "
                "on the workload's context-length distribution and the underlying hardware. "
            )
            reps = max(1, (bench_prompt_token_target * 4) // len(base) + 1)
            prompt = base * reps
            print(f"loader: long-context prompt built, chars={len(prompt)} reps={reps} target_tokens={bench_prompt_token_target}")
        else:
            prompt = "Explain how a transformer attends across heads. " * (bench_input_len // 8)
        bench_state["first_bench_at"] = time.time()
        with open(bench_log, "w") as bf:
            while not stop_flag.is_set():
                try:
                    r = requests.post(
                        f"http://127.0.0.1:{port}/generate",
                        json={
                            "text": [prompt] * bench_batch,
                            "sampling_params": {
                                "temperature": 0.7,
                                "max_new_tokens": bench_output_len,
                            },
                        },
                        timeout=120,
                    )
                    bf.write(f"{time.time():.2f} status={r.status_code} bytes={len(r.content)}\n")
                    bf.flush()
                except Exception as e:
                    bf.write(f"{time.time():.2f} err={e}\n")
                    bf.flush()

    loader = threading.Thread(target=loader_loop, daemon=True)
    loader.start()

    # ---- block until nsys finishes (delay + duration + a little for tail finalize) ----
    expected_s = delay_s + trace_duration_s + 60
    print(f"waiting up to {expected_s}s for nsys to finalize …")
    try:
        rc = server.wait(timeout=expected_s + 600)
        print(f"nsys exited rc={rc}")
    except subprocess.TimeoutExpired:
        print("nsys/server timed out, killing")
        server.kill()
        server.wait()
        rc = -9

    stop_flag.set()
    loader.join(timeout=10)
    log_f.close()

    print("=== bench timing ===")
    print(json.dumps(bench_state, indent=2, default=str))

    # ---- post-process ----
    rep_file = rep_path.with_suffix(".nsys-rep")
    out: dict = {"rep": None, "rc": rc, "bench_state": {k: v for k, v in bench_state.items()}}
    if rep_file.exists():
        sz = rep_file.stat().st_size
        print(f"rep size: {sz/1e6:.1f} MB at {rep_file}")
        out["rep"] = str(rep_file)
        # quick on-host stats so we don't have to download to peek
        stats_cmd = [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum",
            "--format",
            "csv",
            "--output",
            str(rep_dir / "dsv4_decode_stats"),
            str(rep_file),
        ]
        print("stats cmd:", " ".join(stats_cmd))
        s = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=15 * MINUTES)
        print("--- stats stdout (tail) ---")
        print(s.stdout[-4000:])
        print("--- stats stderr (tail) ---")
        print(s.stderr[-2000:])
    else:
        print("WARNING: no .nsys-rep produced")

    trace_vol.commit()
    return out


@app.function(
    image=sglang_image,
    gpu="B200:4",
    timeout=60 * MINUTES,
    volumes={HF_CACHE_PATH: hf_vol, TRACE_PATH: trace_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def sweep_context_lengths(
    targets_k: str = "4,8,16,32,50,100,150,200,250,300,350,400,450,500",
    max_new_tokens: int = 128,
    num_repeats: int = 1,
    max_server_restarts: int = 3,
):
    """Bench-only sweep: launch server, loop over target context lengths, record
    per-step decode latency via streaming, write results to a CSV + txt on the volume.

    No nsys — that would balloon cost and slow things down. Use server log + HTTP
    streaming timestamps to decompose prefill vs decode.
    """
    import json as _json
    import threading
    import requests

    targets = [int(x) * 1000 for x in targets_k.split(",")]
    port = 30000
    rep_dir = Path(TRACE_PATH)
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / "context_sweep.csv"
    txt_path = rep_dir / "context_sweep.txt"
    log_path = rep_dir / "server_sweep.log"

    server_cmd = _sglang_server_cmd(port=port)
    print("server cmd:", " ".join(server_cmd))

    state: dict = {"log_f": None, "server": None, "restarts_used": 0}

    def _spawn_server():
        if state["server"] is not None:
            try:
                state["server"].terminate()
                state["server"].wait(timeout=20)
            except Exception:
                try:
                    state["server"].kill()
                except Exception:
                    pass
        if state["log_f"] is not None:
            try:
                state["log_f"].close()
            except Exception:
                pass
        mode = "a" if log_path.exists() else "w"
        log_f = open(log_path, mode)
        log_f.write(f"\n\n=== server (re)start at {time.time()} restarts_used={state['restarts_used']} ===\n")
        log_f.flush()
        server = subprocess.Popen(server_cmd, stdout=log_f, stderr=subprocess.STDOUT, env=os.environ.copy())
        state["server"] = server
        state["log_f"] = log_f
        print(f"server pid: {server.pid}")
        _wait_for_server(port, timeout_s=20 * MINUTES)
        print("server ready")

    def _is_healthy() -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    _spawn_server()

    results: list[dict] = []
    base = (
        "The hybrid attention mechanism in modern large language models compresses "
        "the key-value cache across the context window to enable efficient decoding. "
        "Researchers have proposed several variants, including compressed sparse attention "
        "and heavily compressed attention, which interleave across transformer layers. "
        "Each variant trades off between recall and compute, and the optimal mix depends "
        "on the workload's context-length distribution and the underlying hardware. "
    )

    # Sampling params used everywhere — temp>0 + top_p guards against the
    # 'probability tensor contains inf/nan' crash we hit with greedy on this build.
    def sp(max_new: int) -> dict:
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": max_new,
            "ignore_eos": True,
        }

    try:
        _wait_for_server(port, timeout_s=20 * MINUTES)
        print("server ready")

        # Probe via /tokenize (no inference, no sampler) for tokens-per-rep.
        # Fall back to a multi-token generate if /tokenize isn't there.
        tokens_per_base = None
        try:
            tr = requests.post(
                f"http://127.0.0.1:{port}/tokenize",
                json={"text": base},
                timeout=30,
            )
            if tr.status_code == 200:
                tj = tr.json()
                ids = tj.get("input_ids") or tj.get("tokens") or tj.get("token_ids")
                if isinstance(ids, list):
                    tokens_per_base = len(ids)
        except Exception:
            pass
        if tokens_per_base is None:
            # robust fallback: generate 4 tokens, read prompt_tokens from meta
            try:
                probe = requests.post(
                    f"http://127.0.0.1:{port}/generate",
                    json={"text": base, "sampling_params": sp(4)},
                    timeout=120,
                ).json()
                pm = probe.get("meta_info", probe)
                tokens_per_base = pm.get("prompt_tokens") or pm.get("input_tokens")
            except Exception as e:
                print("token-count probe failed:", e)
        if not tokens_per_base or tokens_per_base < 8:
            tokens_per_base = 80  # rough default
        print(f"base paragraph encodes to ~{tokens_per_base} tokens per rep")

        for target_tokens in targets:
            for rep_i in range(num_repeats):
                # If a prior crash killed the server, restart (bounded retries)
                if not _is_healthy():
                    if state["restarts_used"] >= max_server_restarts:
                        results.append({
                            "target_k": target_tokens // 1000,
                            "repeat": rep_i,
                            "prompt_tokens": None, "completion_tokens": None,
                            "wall_s": None, "ttft_s": None, "decode_wall_s": None,
                            "decode_tok_per_s": None, "ms_per_decode_step": None,
                            "error": "server dead and max restarts exhausted",
                        })
                        print(f"  target={target_tokens//1000}k rep={rep_i} SKIP — out of restarts")
                        continue
                    print(f"  server unhealthy, restarting before target={target_tokens//1000}k")
                    try:
                        _spawn_server()
                        state["restarts_used"] += 1
                    except Exception as e:
                        results.append({
                            "target_k": target_tokens // 1000,
                            "repeat": rep_i,
                            "prompt_tokens": None, "completion_tokens": None,
                            "wall_s": None, "ttft_s": None, "decode_wall_s": None,
                            "decode_tok_per_s": None, "ms_per_decode_step": None,
                            "error": f"restart failed: {e!r}",
                        })
                        print(f"  target={target_tokens//1000}k rep={rep_i} SKIP — restart failed")
                        continue

                # flush kv cache between trials so each is cold-on-prefix
                try:
                    requests.post(f"http://127.0.0.1:{port}/flush_cache", timeout=30)
                except Exception:
                    pass
                reps = max(1, target_tokens // max(1, tokens_per_base) + 1)
                prompt = base * reps
                t_send = time.perf_counter()
                first_token_t = None
                last_token_t = None
                completion = ""
                tokens_seen = 0
                err: str | None = None
                try:
                    r = requests.post(
                        f"http://127.0.0.1:{port}/generate",
                        json={
                            "text": prompt,
                            "sampling_params": sp(max_new_tokens),
                            "stream": True,
                        },
                        stream=True,
                        timeout=600,
                    )
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: ") :]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            evt = _json.loads(line)
                        except Exception:
                            continue
                        now = time.perf_counter()
                        if first_token_t is None:
                            first_token_t = now
                        last_token_t = now
                        completion = evt.get("text", completion)
                        meta = evt.get("meta_info") or {}
                        if "completion_tokens" in meta:
                            tokens_seen = meta["completion_tokens"]
                        elif "output_token_count" in meta:
                            tokens_seen = meta["output_token_count"]
                    t_end = time.perf_counter()
                except Exception as e:
                    err = repr(e)
                    t_end = time.perf_counter()

                # Authoritative prompt-token count from /tokenize (no inference)
                actual_prompt_tokens = None
                try:
                    tr = requests.post(
                        f"http://127.0.0.1:{port}/tokenize",
                        json={"text": prompt},
                        timeout=60,
                    )
                    if tr.status_code == 200:
                        tj = tr.json()
                        ids = tj.get("input_ids") or tj.get("tokens") or tj.get("token_ids")
                        if isinstance(ids, list):
                            actual_prompt_tokens = len(ids)
                except Exception:
                    pass
                if actual_prompt_tokens is None:
                    actual_prompt_tokens = tokens_per_base * reps  # estimate

                ttft = (first_token_t - t_send) if first_token_t else None
                decode_wall = (last_token_t - first_token_t) if (first_token_t and last_token_t and last_token_t > first_token_t) else None
                tok_per_s = (tokens_seen / decode_wall) if (decode_wall and tokens_seen > 1) else None
                ms_per_step = (decode_wall * 1000 / max(1, tokens_seen - 1)) if (decode_wall and tokens_seen > 1) else None
                row = {
                    "target_k": target_tokens // 1000,
                    "repeat": rep_i,
                    "prompt_tokens": actual_prompt_tokens,
                    "completion_tokens": tokens_seen,
                    "wall_s": round(t_end - t_send, 3),
                    "ttft_s": round(ttft, 3) if ttft is not None else None,
                    "decode_wall_s": round(decode_wall, 3) if decode_wall is not None else None,
                    "decode_tok_per_s": round(tok_per_s, 1) if tok_per_s is not None else None,
                    "ms_per_decode_step": round(ms_per_step, 2) if ms_per_step is not None else None,
                    "error": err,
                }
                results.append(row)
                print(f"  target={target_tokens//1000}k rep={rep_i} -> {row}")

        # write outputs
        import csv as _csv
        cols = ["target_k","repeat","prompt_tokens","completion_tokens","wall_s","ttft_s","decode_wall_s","decode_tok_per_s","ms_per_decode_step","error"]
        with csv_path.open("w") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in results:
                w.writerow(row)

        # human txt
        with txt_path.open("w") as f:
            f.write("DSv4-Flash decode latency sweep — 4x B200, TP=4, sglang 0.5.12\n")
            f.write("targets: " + ", ".join(f"{t//1000}k" for t in targets) + f"  num_repeats={num_repeats}  max_new_tokens={max_new_tokens}\n\n")
            f.write(f"{'target':>7s}  {'rep':>3s}  {'prompt_tok':>10s}  {'compl_tok':>9s}  {'wall_s':>7s}  {'ttft_s':>7s}  {'decode_s':>8s}  {'tok/s':>8s}  {'ms/step':>8s}   error\n")
            f.write("-"*120 + "\n")
            for r in results:
                err = r.get("error")
                err = err if err else ""
                f.write(f"{(str(r['target_k'])+'k'):>7s}  {r['repeat']:>3d}  {str(r['prompt_tokens']):>10s}  {str(r['completion_tokens']):>9s}  {str(r['wall_s']):>7s}  {str(r['ttft_s']):>7s}  {str(r['decode_wall_s']):>8s}  {str(r['decode_tok_per_s']):>8s}  {str(r['ms_per_decode_step']):>8s}   {err}\n")

        print(f"wrote {csv_path} and {txt_path}")
    finally:
        if state["server"] is not None:
            try:
                state["server"].terminate()
                state["server"].wait(timeout=30)
            except Exception:
                try:
                    state["server"].kill()
                except Exception:
                    pass
        if state["log_f"] is not None:
            try:
                state["log_f"].close()
            except Exception:
                pass
        trace_vol.commit()

    return {"n_rows": len(results), "csv": str(csv_path), "txt": str(txt_path)}


@app.function(
    image=sglang_image,
    cpu=4,
    timeout=30 * MINUTES,
    volumes={TRACE_PATH: trace_vol},
)
def analyze_rep(rep_name: str = "dsv4_decode.nsys-rep"):
    """CPU-only re-run of `nsys stats` against an existing rep on the volume.

    Useful when the local nsys is older than the recorder's, since reps are forward-incompatible.
    Writes CSVs back to the same volume.
    """
    rep_dir = Path(TRACE_PATH)
    rep = rep_dir / rep_name
    assert rep.exists(), f"{rep} not found on volume"

    # nuke the stale sqlite so --force-export can rebuild against the new rep
    sqlite_path = rep.with_suffix(".sqlite")
    if sqlite_path.exists():
        print(f"removing stale {sqlite_path}")
        sqlite_path.unlink()

    out_stem = rep_dir / f"{rep.stem}_stats"
    for old in rep_dir.glob(f"{out_stem.name}_*.csv"):
        print(f"removing stale {old}")
        old.unlink()

    reports = [
        "cuda_gpu_kern_sum",
        "cuda_gpu_trace",
        "cuda_api_sum",
        "nvtx_sum",
        "cuda_gpu_mem_time_sum",
        "cuda_gpu_mem_size_sum",
    ]
    cmd = ["nsys", "stats", "--force-export=true", "--format", "csv", "--output", str(out_stem)]
    for r in reports:
        cmd += ["--report", r]
    cmd.append(str(rep))
    print("cmd:", " ".join(cmd))
    s = subprocess.run(cmd, capture_output=True, text=True, timeout=15 * MINUTES)
    print("rc:", s.returncode)
    print("--- stdout (tail) ---")
    print(s.stdout[-6000:])
    print("--- stderr (tail) ---")
    print(s.stderr[-2000:])

    produced = sorted(rep_dir.glob(f"{out_stem.name}*.csv"))
    print("produced:", [str(p) for p in produced])
    trace_vol.commit()
    return [str(p) for p in produced]


@app.local_entrypoint()
def main(stage: str = "sanity"):
    if stage == "sanity":
        print(sanity.remote())
    elif stage == "download":
        print(download_weights.remote())
    elif stage == "trace":
        print(serve_and_trace.remote())
    elif stage == "dry":
        print(serve_and_trace.remote(dry_run=True))
    elif stage == "analyze":
        print(analyze_rep.remote())
    elif stage == "trace_100k":
        print(
            serve_and_trace.remote(
                trace_duration_s=25,
                delay_s=540,
                bench_batch=2,
                bench_output_len=128,
                bench_prompt_token_target=100_000,
                output_tag="100k",
            )
        )
    elif stage == "analyze_100k":
        print(analyze_rep.remote("dsv4_decode_100k.nsys-rep"))
    elif stage == "sweep":
        print(sweep_context_lengths.remote())
    else:
        raise ValueError(f"unknown stage: {stage}")
