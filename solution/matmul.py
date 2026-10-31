import torch
import triton
import triton.language as tl


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def fused_linear_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + rows[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=(rows[:, None] < M) & (k_off[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_off[:, None] * stride_bk + cols[None, :] * stride_bn,
            mask=(k_off[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(
        c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        c,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def fused_linear_transposed_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + rows[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=(rows[:, None] < M) & (k_off[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + cols[None, :] * stride_bn + k_off[:, None] * stride_bk,
            mask=(cols[None, :] < N) & (k_off[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    tl.store(
        c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def _launch(a, b, shape, transposed_b=False):
    m = shape[0]
    k = shape[1]
    n = shape[2]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    if transposed_b:
        fused_linear_transposed_kernel[grid](
            a, b, out, m, k, n,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    else:
        fused_linear_kernel[grid](
            a, b, out, m, k, n,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    return out


class FusedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        original_shape = x.shape
        x = x.contiguous().view(-1, weight.shape[1])
        weight = weight.contiguous()
        ctx.original_shape = original_shape
        ctx.flat_shape = x.shape
        ctx.save_for_backward(x, weight)
        ctx.bias = bias
        out = _launch(x, weight, (x.shape[0], x.shape[1], weight.shape[0]), transposed_b=True)
        if bias is not None:
            out = out + bias
        return out.view(*original_shape[:-1], weight.shape[0])


    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad = grad_output.contiguous().view(-1, weight.shape[0])
        grad_x = _launch(grad, weight, (grad.shape[0], weight.shape[0], weight.shape[1]))
        grad_weight = _launch(
            grad.transpose(0, 1),
            x,
            (weight.shape[0], grad.shape[0], weight.shape[1]),
        )
        grad_bias = grad.sum(0) if ctx.bias is not None else None
        return grad_x.view(*ctx.original_shape), grad_weight, grad_bias


def fused_linear(x, weight, bias=None):
    return FusedLinear.apply(x, weight, bias)


def patch_linear():
    if getattr(torch.nn.Linear.forward, "_fused_matmul_patch", False):
        return
    original = torch.nn.Linear.forward

    def forward(self, input_tensor):
        if (
            input_tensor.is_cuda
            and input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and input_tensor.numel() != 0
            and input_tensor.shape[-1] == self.in_features
        ):
            return fused_linear(input_tensor, self.weight, self.bias)
        return original(self, input_tensor)

    forward._fused_matmul_patch = True
    torch.nn.Linear.forward = forward
