import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_loss_grad_entropy_argmax_match_pytorch(dtype):
    from solution.vocab_loss import _STASH, fused_for_causal_lm_loss

    torch.manual_seed(0)
    batch, tokens, vocab = 3, 37, 5000
    logits = (torch.randn(batch, tokens, vocab, device="cuda") * 3).to(dtype).requires_grad_()
    labels = torch.randint(0, vocab, (batch, tokens), device="cuda")
    labels[:, :9] = -100
    labels[1, 20:] = -100
    items = (labels[:, 1:] != -100).sum()

    ref_logits = logits.detach().clone().requires_grad_()
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
    ref = F.cross_entropy(ref_logits.float().view(-1, vocab), shifted.reshape(-1),
                          ignore_index=-100, reduction="sum") / items
    ref.backward()

    out = fused_for_causal_lm_loss(logits, labels, vocab, num_items_in_batch=items)
    out.backward()
    tol = 2e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(logits.grad.float(), ref_logits.grad.float(), rtol=1e-2, atol=1e-4)

    logp = F.log_softmax(logits.detach().float(), dim=-1)
    ref_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(_STASH["ent"].view(batch, tokens), ref_entropy, rtol=1e-3, atol=1e-3)
    best = logits.detach().float().argmax(-1)
    picked = logits.detach().float().gather(-1, _STASH["amax"].view(batch, tokens, 1).long())
    torch.testing.assert_close(picked.squeeze(-1), logits.detach().float().max(-1).values)
    assert (best == _STASH["amax"].view(batch, tokens)).float().mean() > 0.99
