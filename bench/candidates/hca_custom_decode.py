"""HCA decode candidate — placeholder for the SM100 kernel binding.

Returns zeros until the torch op `dsv4::hca::decode_fwd` is wired up via
`hca/setup.py` (companion to the existing `hca_compress_reduce_fwd`).
The bench runner will report a numerical mismatch — that's expected
until the real kernel call is plugged in below.

Interface:
    candidate(inputs: HCADecodeInputs, *, spec: HCADecodeSpec) -> torch.Tensor
        returns O [B, n_h, c]  bf16
"""
import torch

NAME = "hca_custom_decode"
VARIANT = "hca_decode"


def candidate(inputs, *, spec):
    # TODO: replace with:
    #   import dsv4_hca
    #   return dsv4_hca.decode_fwd(
    #       inputs.Q,
    #       inputs.Kc, inputs.Kc_scales, inputs.Kc_rope,
    #       inputs.Kswa, inputs.Kswa_scales, inputs.Kswa_rope,
    #       inputs.sink_logits, inputs.sm_scale,
    #       spec.M_cur, spec.swa_len,
    #   )
    return torch.zeros(spec.B, spec.n_h, spec.c,
                       dtype=torch.bfloat16, device=inputs.Q.device)
