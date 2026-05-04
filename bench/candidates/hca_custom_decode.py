"""HCA decode candidate — calls the SM100 kernel via ctypes.

Resolution order for the .so:
  1. $DSV4_HCA_SO  — explicit path to a pre-built libhca.so
  2. <hca>/build/libhca.so if it already exists
  3. Run `python <hca>/compile.py` to build it (needs nvcc on PATH)
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import torch

NAME = "hca_custom_decode"
VARIANT = "hca_decode"


_LIB = None
_FN  = None


def _hca_root() -> Path:
    return Path(os.environ.get(
        "DSV4_HCA_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "hca"),
    ))


def _resolve_so() -> Path:
    explicit = os.environ.get("DSV4_HCA_SO")
    if explicit:
        return Path(explicit)
    so = _hca_root() / "build" / "libhca.so"
    if so.exists():
        return so
    rc = subprocess.run(
        ["python", str(_hca_root() / "compile.py")],
        check=False,
    ).returncode
    if rc != 0 or not so.exists():
        raise RuntimeError(f"compile.py failed (rc={rc}); no {so}")
    return so


def _load():
    global _LIB, _FN
    if _LIB is not None:
        return
    _LIB = ctypes.CDLL(str(_resolve_so()))
    fn = _LIB.hca_decode_fwd
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,  # Q
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # Kc, Kc_s, Kc_rope
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # Kswa, Kswa_s, Kswa_rope
        ctypes.c_void_p,  # sink
        ctypes.c_void_p,  # O
        ctypes.c_void_p, ctypes.c_void_p,  # partial_O, partial_lse
        ctypes.c_float,   # sm_scale
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # B, n_h, c, n_rope
        ctypes.c_int, ctypes.c_int,  # M_cur, swa_len
        ctypes.c_void_p,  # stream
    ]
    _FN = fn


def candidate(inputs, *, spec):
    _load()

    Q = inputs.Q.contiguous()
    Kc = inputs.Kc.contiguous()
    if Kc.dtype == torch.float8_e4m3fn:
        Kc = Kc.view(torch.uint8)
    Kc_s = inputs.Kc_scales.contiguous()
    Kc_rope = inputs.Kc_rope.contiguous()
    Kswa = inputs.Kswa.contiguous()
    if Kswa.dtype == torch.float8_e4m3fn:
        Kswa = Kswa.view(torch.uint8)
    Kswa_s = inputs.Kswa_scales.contiguous()
    Kswa_rope = inputs.Kswa_rope.contiguous()
    sink = inputs.sink_logits.contiguous() if inputs.sink_logits is not None else None

    B, n_h, dq = Q.shape
    c = Kc.shape[-1]
    n_rope = dq - c
    M_cur = int(spec.M_cur)
    swa_len = int(spec.swa_len)

    O = torch.empty(B, n_h, c, dtype=torch.bfloat16, device=Q.device)
    partial_O   = torch.empty(1, B, n_h, c, dtype=torch.float32, device=Q.device)
    partial_lse = torch.empty(1, B, n_h,    dtype=torch.float32, device=Q.device)

    stream = torch.cuda.current_stream(Q.device).cuda_stream

    rc = _FN(
        Q.data_ptr(),
        Kc.data_ptr(), Kc_s.data_ptr(), Kc_rope.data_ptr(),
        Kswa.data_ptr(), Kswa_s.data_ptr(), Kswa_rope.data_ptr(),
        sink.data_ptr() if sink is not None else 0,
        O.data_ptr(),
        partial_O.data_ptr(), partial_lse.data_ptr(),
        ctypes.c_float(float(inputs.sm_scale)),
        B, n_h, c, n_rope,
        M_cur, swa_len,
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        raise RuntimeError(f"hca_decode_fwd cudaError={rc}")
    return O
