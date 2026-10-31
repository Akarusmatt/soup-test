"""Fused RMSNorm (forward + backward) in Triton for Qwen2RMSNorm.

Why this kernel
---------------
prepare_model_for_kbit_training keeps all norm weights in fp32, so the stock
Qwen2RMSNorm materialises several fp32 [B, T, H] tensors per call (x.float(),
pow(2), rsqrt product, cast back, weight multiply) and autograd keeps them alive
for backward. Qwen2.5-1.5B has 28 * 2 + 1 = 57 such norms and no gradient
checkpointing is used, so this is both element-wise/cast time and saved memory.

This kernel reads x once, keeps only x and the per-row rstd for backward, and
writes the output once. Backward recomputes everything from x, rstd and weight.

Numerics follow the reference exactly where it matters:
    n   = (x.float() * rstd).to(x.dtype)       # rounded to the input dtype
    out = weight * n                           # promoted dtype (fp32 weight -> fp32)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X, W, Y, RSTD,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    tl.store(RSTD + row, rstd)

    # round to the input dtype first, like `hidden.to(input_dtype)` in the reference
    n = (x * rstd).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, (w * n).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    DY, X, W, RSTD, DX,
    stride_dy, stride_x, stride_dx,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)

    dn = dy * w
    # dx = rstd * dn - x * rstd^3 / N * sum(dn * x)
    c = tl.sum(dn * x, axis=0) / N
    dx = rstd * dn - x * (rstd * rstd * rstd) * c
    tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        rows, n = x2.shape

        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty((rows, n), device=x.device, dtype=out_dtype)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)

        block = triton.next_power_of_2(n)
        num_warps = 4 if block <= 2048 else 8
        _rmsnorm_fwd_kernel[(rows,)](
            x2, weight, y, rstd,
            x2.stride(0), y.stride(0),
            n, eps,
            BLOCK=block, num_warps=num_warps,
        )
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        ctx.x_dtype = x.dtype
        ctx.block = block
        ctx.num_warps = num_warps
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, n = x2.shape
        dy2 = dy.reshape(-1, n)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty((rows, n), device=x2.device, dtype=ctx.x_dtype)
        _rmsnorm_bwd_kernel[(rows,)](
            dy2, x2, weight, rstd, dx,
            dy2.stride(0), x2.stride(0), dx.stride(0),
            n,
            BLOCK=ctx.block, num_warps=ctx.num_warps,
        )
        return dx.reshape(ctx.shape), None, None


def fused_rmsnorm(x, weight, eps):
    return _FusedRMSNorm.apply(x, weight, eps)


def _forward(self, hidden_states):
    weight = self.weight
    eps = getattr(self, "variance_epsilon", None)
    if eps is None:
        eps = self.eps
    # Fallback to the original path whenever the fast path does not apply:
    # CPU tensors (Soup regression tests use fakes), a trainable norm weight
    # (the kernel does not produce dweight), or an unsupported hidden size.
    if (
        not isinstance(hidden_states, torch.Tensor)
        or not hidden_states.is_cuda
        or weight.requires_grad
        or hidden_states.shape[-1] > 16384
    ):
        return _ORIG_FORWARD(self, hidden_states)
    return fused_rmsnorm(hidden_states, weight, eps)


_ORIG_FORWARD = None


def patch():
    """Replace Qwen2RMSNorm.forward. Safe to call more than once."""
    global _ORIG_FORWARD
    from transformers.models.qwen2 import modeling_qwen2 as m

    cls = m.Qwen2RMSNorm
    if getattr(cls.forward, "_soup_fused_rmsnorm", False):
        return
    _ORIG_FORWARD = cls.forward
    _forward._soup_fused_rmsnorm = True
    cls.forward = _forward
