"""One pass over the [tokens, 151936] logits instead of ~8.

Stock path per training step (Qwen2.5-1.5B, batch 4 x 512, fp16 logits):
  * transformers ForCausalLMLoss: logits.float() copy, log_softmax output, fp32 grad
  * TRL compute_loss: entropy_from_logits (chunked log_softmax / exp), logits[..., :-1, :]
    .contiguous() copy, argmax
  * Soup rewind recorder: row_losses() = one cross_entropy per row
Every one of them materialises or re-reads a full-vocabulary tensor.

Here a single Triton kernel reads each logits row once (online softmax) and produces
logsumexp, entropy, argmax and the target-token loss. A second kernel writes the
cross-entropy gradient directly in the logits dtype. The statistics are kept in a small
side cache keyed on the logits tensor, so TRL's entropy / token accuracy and Soup's
per-row losses are read from it instead of being recomputed from the logits.
"""

import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

IGNORE = -100
BLOCK = 2048

_STASH = {}
_PENDING = {}


@triton.jit
def _vocab_stats_kernel(
    LOGITS, TARGETS, LSE, ENT, AMAX, LOSS,
    V, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = LOGITS + row * stride
    offs = tl.arange(0, BLOCK)

    m_v = tl.full([BLOCK], float("-inf"), tl.float32)
    s_v = tl.zeros([BLOCK], tl.float32)
    t_v = tl.zeros([BLOCK], tl.float32)
    best = tl.full([BLOCK], float("-inf"), tl.float32)
    best_i = tl.zeros([BLOCK], tl.int32)

    for start in range(0, V, BLOCK):
        cols = start + offs
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        new_m = tl.maximum(m_v, x)
        finite = new_m > float("-inf")
        alpha = tl.where(finite, tl.exp(m_v - new_m), 1.0)
        p = tl.where(finite, tl.exp(x - new_m), 0.0)
        s_v = s_v * alpha + p
        t_v = t_v * alpha + p * tl.where(mask, x, 0.0)
        m_v = new_m
        better = x > best
        best = tl.where(better, x, best)
        best_i = tl.where(better, cols, best_i)

    top = tl.max(m_v, axis=0)
    w = tl.exp(m_v - top)
    s = tl.sum(s_v * w, axis=0)
    t = tl.sum(t_v * w, axis=0)
    lse = top + tl.log(s)
    ent = lse - t / s

    gmax = tl.max(best, axis=0)
    arg = tl.min(tl.where(best == gmax, best_i, 2147483647), axis=0)

    tgt = tl.load(TARGETS + row)
    valid = tgt != -100
    xt = tl.load(base + tl.where(valid, tgt, 0)).to(tl.float32)
    loss = tl.where(valid, lse - xt, 0.0)

    tl.store(LSE + row, lse)
    tl.store(ENT + row, ent)
    tl.store(AMAX + row, arg)
    tl.store(LOSS + row, loss)


@triton.jit
def _vocab_grad_kernel(
    LOGITS, TARGETS, LSE, G, DX,
    V, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    tgt = tl.load(TARGETS + row)
    lse = tl.load(LSE + row)
    g = tl.load(G + row).to(tl.float32)
    scale = tl.where(tgt != -100, g, 0.0)
    in_base = LOGITS + row * stride
    out_base = DX + row * stride

    for start in range(0, V, BLOCK):
        cols = start + offs
        mask = cols < V
        x = tl.load(in_base + cols, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        d = (p - tl.where(cols == tgt, 1.0, 0.0)) * scale
        tl.store(out_base + cols, d.to(DX.dtype.element_ty), mask=mask)


def _launch_stats(logits2d, targets):
    rows, vocab = logits2d.shape
    dev = logits2d.device
    lse = torch.empty(rows, device=dev, dtype=torch.float32)
    ent = torch.empty(rows, device=dev, dtype=torch.float32)
    amax = torch.empty(rows, device=dev, dtype=torch.int32)
    loss = torch.empty(rows, device=dev, dtype=torch.float32)
    _vocab_stats_kernel[(rows,)](
        logits2d, targets, lse, ent, amax, loss,
        vocab, logits2d.stride(0),
        BLOCK=BLOCK, num_warps=8,
    )
    return lse, ent, amax, loss


class _FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits2d, targets):
        lse, ent, amax, loss = _launch_stats(logits2d, targets)
        _PENDING["ent"] = ent
        _PENDING["amax"] = amax
        ctx.save_for_backward(logits2d, targets, lse)
        return loss

    @staticmethod
    def backward(ctx, grad_rows):
        logits2d, targets, lse = ctx.saved_tensors
        grad_rows = grad_rows.to(torch.float32).contiguous()
        dx = torch.empty_like(logits2d)
        _vocab_grad_kernel[(logits2d.shape[0],)](
            logits2d, targets, lse, grad_rows, dx,
            logits2d.shape[1], logits2d.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return dx, None


def _remember(logits, loss_rows):
    _STASH.clear()
    _STASH.update(
        ref=weakref.ref(logits),
        shape=tuple(logits.shape),
        version=logits._version,
        ent=_PENDING.pop("ent"),
        amax=_PENDING.pop("amax"),
        loss=loss_rows.detach(),
    )


def _hit(logits):
    ref = _STASH.get("ref")
    return (
        ref is not None
        and ref() is logits
        and _STASH["shape"] == tuple(logits.shape)
        and _STASH["version"] == logits._version
    )


def token_stats(logits):
    """(entropy, argmax) of every position of a [..., V] CUDA logits tensor, shaped like
    logits.shape[:-1]. Served from the loss kernel's cache when it saw this tensor."""
    lead = logits.shape[:-1]
    if _hit(logits):
        return _STASH["ent"].view(lead), _STASH["amax"].view(lead)
    logits2d = logits.reshape(-1, logits.shape[-1])
    if logits2d.stride(-1) != 1:
        logits2d = logits2d.contiguous()
    targets = torch.full((logits2d.shape[0],), IGNORE, device=logits.device, dtype=torch.long)
    with torch.no_grad():
        _, ent, amax, _ = _launch_stats(logits2d, targets)
    return ent.view(lead), amax.view(lead)


def fused_for_causal_lm_loss(
    logits, labels, vocab_size, num_items_in_batch=None, ignore_index=IGNORE,
    shift_labels=None, **kwargs,
):
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_cuda
        or logits.dim() != 3
        or logits.shape[-1] != vocab_size
        or ignore_index != IGNORE
    ):
        return _ORIG_LOSS(
            logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
            ignore_index=ignore_index, shift_labels=shift_labels, **kwargs,
        )
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    targets = shift_labels.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
    logits2d = logits.reshape(-1, vocab_size)
    if logits2d.stride(-1) != 1:
        logits2d = logits2d.contiguous()

    rows = _FusedCrossEntropy.apply(logits2d, targets)
    _remember(logits, rows)

    total = rows.sum()
    if num_items_in_batch is None:
        return total / (targets != IGNORE).sum().clamp(min=1)
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(total.device)
    return total / num_items_in_batch


def fused_entropy_from_logits(logits, chunk_size=128):
    if not isinstance(logits, torch.Tensor) or not logits.is_cuda:
        return _ORIG_ENTROPY(logits, chunk_size)
    return token_stats(logits)[0]


def fast_row_losses(logits, labels):
    if isinstance(logits, torch.Tensor) and logits.is_cuda and _hit(logits):
        from soup_cli.trainer.rewind_hf import row_losses_from_token_losses

        return row_losses_from_token_losses(_STASH["loss"].view(logits.shape[:-1]), labels)
    return _ORIG_ROW_LOSSES(logits, labels)


def _make_compute_loss(original, sft_cls):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (
            not isinstance(model, torch.nn.Module)
            or getattr(self.args, "use_liger_kernel", False)
            or "shift_labels" in inputs
            or "labels" not in inputs
            or getattr(self, "num_virtual_tokens", 0) > 0
            or getattr(self, "aux_loss_enabled", False)
        ):
            return original(self, model, inputs, return_outputs, num_items_in_batch)

        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False

        loss, outputs = super(sft_cls, self).compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        logits = outputs.logits
        with torch.no_grad():
            entropy_rows, argmax_rows = token_stats(logits)
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"]
                entropy = torch.sum(entropy_rows * mask) / mask.sum()
            elif "position_ids" in inputs:
                entropy = torch.mean(entropy_rows)
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens = self.accelerator.gather_for_metrics(local).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        with torch.no_grad():
            shift_labels = labels[..., 1:]
            predictions = argmax_rows[..., :-1].to(shift_labels.dtype)
            valid = shift_labels != -100
            correct = (predictions == shift_labels) & valid
            correct_tokens = self.accelerator.gather_for_metrics(correct.sum())
            total_tokens = self.accelerator.gather_for_metrics(valid.sum())
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_fused_vocab = True
    return compute_loss


_ORIG_LOSS = None
_ORIG_ENTROPY = None
_ORIG_ROW_LOSSES = None


def patch_vocab_loss():
    global _ORIG_LOSS, _ORIG_ENTROPY, _ORIG_ROW_LOSSES
    from transformers.loss import loss_utils

    if getattr(loss_utils.LOSS_MAPPING["ForCausalLM"], "_soup_fused_vocab", False):
        return
    _ORIG_LOSS = loss_utils.ForCausalLMLoss
    fused_for_causal_lm_loss._soup_fused_vocab = True
    loss_utils.LOSS_MAPPING["ForCausalLM"] = fused_for_causal_lm_loss
    loss_utils.ForCausalLMLoss = fused_for_causal_lm_loss

    try:
        import trl
        from trl.trainer import sft_trainer
    except Exception:  # noqa: BLE001
        return
    _ORIG_ENTROPY = sft_trainer.entropy_from_logits
    sft_trainer.entropy_from_logits = fused_entropy_from_logits

    if getattr(trl, "__version__", "") == "0.29.1" and not getattr(
        sft_trainer.SFTTrainer.compute_loss, "_soup_fused_vocab", False
    ):
        sft_trainer.SFTTrainer.compute_loss = _make_compute_loss(
            sft_trainer.SFTTrainer.compute_loss, sft_trainer.SFTTrainer
        )

    try:
        from soup_cli.trainer import rewind_hf
    except Exception:  # noqa: BLE001
        return
    _ORIG_ROW_LOSSES = rewind_hf.row_losses
    rewind_hf.row_losses = fast_row_losses
