import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def rmsnorm_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    rows, hidden, eps,
    x_stride_row, output_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mean_square = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, hidden, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < hidden
        x = tl.load(x_ptr + row * x_stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean_square += tl.where(mask, x * x, 0.0)
    mean_square = tl.sum(mean_square, axis=0) / hidden
    rstd = tl.rsqrt(mean_square + eps)

    for start in range(0, hidden, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < hidden
        x = tl.load(x_ptr + row * x_stride_row + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        tl.store(
            output_ptr + row * output_stride_row + cols,
            (x * rstd * weight).to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def rmsnorm_backward_kernel(
    grad_ptr, x_ptr, weight_ptr, output_ptr,
    rows, hidden, eps,
    grad_stride_row, x_stride_row, output_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    square_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    projection = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, hidden, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < hidden
        x = tl.load(x_ptr + row * x_stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        square_sum += tl.where(mask, x * x, 0.0)
    rstd = tl.rsqrt(tl.sum(square_sum, axis=0) / hidden + eps)

    for start in range(0, hidden, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < hidden
        x = tl.load(x_ptr + row * x_stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + row * grad_stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        weighted_grad = grad * weight
        projection += tl.where(mask, weighted_grad * x * rstd, 0.0)
    covariance = tl.sum(projection, axis=0) / hidden

    for start in range(0, hidden, BLOCK_SIZE):
        cols = start + offsets
        mask = cols < hidden
        x = tl.load(x_ptr + row * x_stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + row * grad_stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dx = rstd * (grad * weight - x * rstd * covariance)
        tl.store(
            output_ptr + row * output_stride_row + cols,
            dx.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        flat_x = x.contiguous().view(-1, x.shape[-1])
        output = torch.empty_like(flat_x)
        rows = flat_x.shape[0]
        rmsnorm_forward_kernel[(rows,)](
            flat_x, weight, output,
            rows, flat_x.shape[1], eps,
            flat_x.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(flat_x, weight)
        ctx.eps = eps
        ctx.original_shape = x.shape
        return output.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, x.shape[-1])
        grad_x = torch.empty_like(x)
        rmsnorm_backward_kernel[(x.shape[0],)](
            grad_output, x, weight, grad_x,
            x.shape[0], x.shape[-1], ctx.eps,
            grad_output.stride(0), x.stride(0), grad_x.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        normalized = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ctx.eps)
        grad_weight = (grad_output.float() * normalized.float()).sum(dim=0).to(weight.dtype)
        return grad_x.view(ctx.original_shape), grad_weight, None


def patch_qwen2_rms_norm():
    from transformers.models.qwen2 import modeling_qwen2

    if getattr(modeling_qwen2.Qwen2RMSNorm.forward, "_fused_rmsnorm_patch", False):
        return
    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if hidden_states.is_cuda and hidden_states.shape[-1] == self.weight.shape[0]:
            return FusedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
        return original(hidden_states)

    forward._fused_rmsnorm_patch = True
    modeling_qwen2.Qwen2RMSNorm.forward = forward
