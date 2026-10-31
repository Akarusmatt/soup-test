import pytest
import torch

from solution.rmsnorm import fused_rmsnorm

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _ref(x, w, eps):
    dt = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dt)


@cuda
@pytest.mark.parametrize("hidden", [1536, 1000, 2048])
def test_forward_backward_matches_reference(hidden):
    torch.manual_seed(0)
    x = torch.randn(4, 512, hidden, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden, device="cuda", dtype=torch.float32)
    eps = 1e-6

    x_ref = x.clone().requires_grad_()
    x_new = x.clone().requires_grad_()
    dy = torch.randn(4, 512, hidden, device="cuda", dtype=torch.float32)

    y_ref = _ref(x_ref, w, eps)
    y_new = fused_rmsnorm(x_new, w, eps)
    assert y_new.dtype == y_ref.dtype
    torch.testing.assert_close(y_new, y_ref, rtol=2e-3, atol=2e-3)

    y_ref.backward(dy)
    y_new.backward(dy)
    torch.testing.assert_close(x_new.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
