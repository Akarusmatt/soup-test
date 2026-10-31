import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1536, 1536), (256, 1536), (77, 130)])
def test_nf4_dequant_matches_reference(dtype, shape):
    import bitsandbytes.functional as F

    from solution.nf4_dequant import _code, nf4_dequantize

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(weight, blocksize=64, quant_type="nf4")
    flat = packed.reshape(-1)
    code = _code(weight.device)

    nibbles = torch.empty(flat.numel() * 2, dtype=torch.long, device="cuda")
    nibbles[0::2] = (flat >> 4).long()
    nibbles[1::2] = (flat & 15).long()
    n = weight.numel()
    values = code[nibbles][:n]
    scale = state.absmax.float().repeat_interleave(64)[:n]
    ref = (values * scale).to(dtype).view(shape)

    out = torch.empty(shape, device="cuda", dtype=dtype)
    nf4_dequantize(flat.contiguous(), state.absmax.float().contiguous(), code, 64, out)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
