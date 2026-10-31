"""NF4 weight dequantisation in Triton, hooked into bitsandbytes' CUDA backend.

Every 4-bit Linear dequantises its weight twice per step (forward and the grad-input matmul of
backward). This kernel reproduces bitsandbytes' result exactly: the packed byte j holds output
element 2j in its high nibble and 2j+1 in its low nibble, and each value is
code[nibble] * absmax[element // blocksize] in fp32, cast once to the output dtype.
"""

import torch
import triton
import triton.language as tl

BYTES_PER_PROGRAM = 1024


@triton.jit
def _nf4_dequant_kernel(
    A, ABSMAX, CODE, OUT,
    n_bytes, n_out,
    BS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_bytes
    packed = tl.load(A + idx, mask=mask, other=0).to(tl.int32)
    scale = tl.load(ABSMAX + (idx * 2) // BS, mask=mask, other=0.0)
    hi = tl.load(CODE + (packed >> 4)) * scale
    lo = tl.load(CODE + (packed & 15)) * scale
    dtype = OUT.dtype.element_ty
    tl.store(OUT + idx * 2, hi.to(dtype), mask=mask & (idx * 2 < n_out))
    tl.store(OUT + idx * 2 + 1, lo.to(dtype), mask=mask & (idx * 2 + 1 < n_out))


def nf4_dequantize(packed, absmax, code, blocksize, out):
    n_out = out.numel()
    n_bytes = (n_out + 1) // 2
    grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
    _nf4_dequant_kernel[grid](
        packed, absmax, code, out, n_bytes, n_out,
        BS=blocksize, BLOCK=BYTES_PER_PROGRAM, num_warps=4,
    )
    return out


_CODES = {}


def _code(device):
    key = str(device)
    if key not in _CODES:
        from bitsandbytes.functional import get_4bit_type

        _CODES[key] = get_4bit_type("nf4", device=device).to(torch.float32).contiguous()
    return _CODES[key]


def patch_nf4_dequant():
    try:
        from bitsandbytes.backends.cuda import ops
    except Exception:  # noqa: BLE001
        return
    original = getattr(ops, "_dequantize_4bit_impl", None)
    if original is None or getattr(original, "_soup_triton", False):
        return

    def impl(A, absmax, blocksize, quant_type, dtype, out):
        if (
            quant_type != "nf4"
            or not out.is_cuda
            or not out.is_contiguous()
            or absmax.dtype != torch.float32
            or dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return original(A, absmax, blocksize, quant_type, dtype, out)
        if A.dtype != torch.uint8:
            A = A.contiguous().view(torch.uint8)
        nf4_dequantize(A.contiguous(), absmax.contiguous(), _code(out.device), blocksize, out)

    impl._soup_triton = True
    ops._dequantize_4bit_impl = impl
