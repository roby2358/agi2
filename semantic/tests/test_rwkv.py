"""Tests for the RWKV model."""

import pytest
import torch
from src.config import AGI2Config
from src.model import AGI2Model
from src.rwkv import RWKVBlock, RWKVChannelMix, RWKVModel, RWKVTimeMix


def tiny_config() -> AGI2Config:
    return AGI2Config(
        vocab_size=100,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )


@pytest.mark.unit
class TestRWKVTimeMix:
    def test_forward_shape(self):
        mix = RWKVTimeMix(d_model=32, layer_id=0, n_layer=2)
        x = torch.randn(2, 10, 32)
        assert mix(x).shape == (2, 10, 32)

    def test_wkv_stability_with_extreme_keys(self):
        mix = RWKVTimeMix(d_model=8, layer_id=0, n_layer=2)
        k = torch.tensor([[[100.0] * 8, [-100.0] * 8, [0.0] * 8]])
        v = torch.randn(1, 3, 8)
        out = mix._wkv(k, v)
        assert torch.isfinite(out).all()

    def test_first_output_is_first_value(self):
        # With no history, WKV at t=0 reduces to the first value vector.
        mix = RWKVTimeMix(d_model=8, layer_id=0, n_layer=2)
        k = torch.randn(1, 3, 8)
        v = torch.randn(1, 3, 8)
        out = mix._wkv(k, v)
        assert torch.allclose(out[:, 0], v[:, 0], atol=1e-5)

    @pytest.mark.parametrize("seq_len", [1, 7, 16, 30, 32, 33, 100])
    def test_chunked_matches_loop(self, seq_len):
        # The chunked scan must reproduce the per-step reference across
        # sequence lengths shorter than, equal to, and straddling the
        # chunk size (including a non-multiple remainder chunk).
        torch.manual_seed(0)
        mix = RWKVTimeMix(d_model=16, layer_id=1, n_layer=4)
        k = torch.randn(2, seq_len, 16) * 3
        v = torch.randn(2, seq_len, 16)
        assert torch.allclose(mix._wkv(k, v), mix._wkv_loop(k, v), atol=1e-5, rtol=1e-4)

    def test_chunked_matches_loop_extreme_inputs(self):
        # Fast-decay channels and huge key magnitudes stress the log-space
        # stabilization; both implementations must agree and stay finite.
        torch.manual_seed(1)
        mix = RWKVTimeMix(d_model=8, layer_id=0, n_layer=2)
        with torch.no_grad():
            mix.time_decay.copy_(
                torch.linspace(-5.0, 3.0, 8)
            )  # decay e^-5 .. e^3 per step
        k = torch.randn(2, 40, 8) * 50
        v = torch.randn(2, 40, 8)
        chunked = mix._wkv(k, v)
        loop = mix._wkv_loop(k, v)
        assert torch.isfinite(chunked).all()
        assert torch.allclose(chunked, loop, atol=1e-4, rtol=1e-3)

    def test_chunked_matches_loop_gradients(self):
        # Training runs through the chunked path, so its gradients must
        # match the reference, for inputs and the time parameters alike.
        torch.manual_seed(2)
        mix = RWKVTimeMix(d_model=8, layer_id=0, n_layer=2)
        k = torch.randn(1, 35, 8, requires_grad=True)
        v = torch.randn(1, 35, 8, requires_grad=True)

        grads = {}
        for name, fn in [("chunked", mix._wkv), ("loop", mix._wkv_loop)]:
            mix.zero_grad()
            if k.grad is not None:
                k.grad = None
                v.grad = None
            fn(k, v).square().sum().backward()
            grads[name] = (
                k.grad.clone(),
                v.grad.clone(),
                mix.time_decay.grad.clone(),
                mix.time_first.grad.clone(),
            )

        for got, want in zip(grads["chunked"], grads["loop"]):
            assert torch.isfinite(got).all()
            assert torch.allclose(got, want, atol=1e-4, rtol=1e-3)


@pytest.mark.unit
class TestRWKVChannelMix:
    def test_forward_shape(self):
        mix = RWKVChannelMix(d_model=32, d_ff=128, layer_id=0, n_layer=2)
        x = torch.randn(2, 10, 32)
        assert mix(x).shape == (2, 10, 32)


@pytest.mark.unit
class TestRWKVBlock:
    def test_forward_shape(self):
        block = RWKVBlock(d_model=32, d_ff=128, layer_id=0, n_layer=2)
        x = torch.randn(2, 10, 32)
        assert block(x).shape == (2, 10, 32)


@pytest.mark.unit
class TestRWKVModel:
    def test_forward_shape(self):
        model = RWKVModel(tiny_config())
        input_ids = torch.randint(0, 100, (2, 12))
        logits = model(input_ids)
        assert logits.shape == (2, 12, 100)

    def test_forward_hidden(self):
        model = RWKVModel(tiny_config())
        input_ids = torch.randint(0, 100, (2, 12))
        logits, hidden = model.forward_hidden(input_ids)
        assert logits.shape == (2, 12, 100)
        assert hidden.shape == (2, 12, 32)

    def test_causality(self):
        # Changing a future token must not change earlier outputs.
        model = RWKVModel(tiny_config())
        model.eval()
        input_ids = torch.randint(0, 100, (1, 16))
        modified = input_ids.clone()
        modified[0, 10] = (modified[0, 10] + 1) % 100
        with torch.no_grad():
            out_a = model(input_ids)
            out_b = model(modified)
        assert torch.allclose(out_a[:, :10], out_b[:, :10], atol=1e-5)
        assert not torch.allclose(out_a[:, 10:], out_b[:, 10:], atol=1e-5)

    def test_backward_pass(self):
        model = RWKVModel(tiny_config())
        input_ids = torch.randint(0, 100, (2, 8))
        logits = model(input_ids)
        logits.sum().backward()
        grads = [p.grad for p in model.parameters()]
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)

    def test_no_position_length_limit(self):
        # RWKV has no position embeddings, so sequences longer than
        # n_positions still work.
        model = RWKVModel(tiny_config())
        input_ids = torch.randint(0, 100, (1, 64))  # n_positions is 32
        assert model(input_ids).shape == (1, 64, 100)

    def test_parameter_count_close_to_transformer(self):
        config = tiny_config()
        rwkv_params = RWKVModel(config).get_num_params()
        gpt_params = AGI2Model(config).get_num_params()
        assert abs(rwkv_params - gpt_params) / gpt_params < 0.2

    def test_state_dict_roundtrip(self):
        config = tiny_config()
        model_a = RWKVModel(config)
        model_b = RWKVModel(config)
        model_b.load_state_dict(model_a.state_dict())
        model_a.eval()
        model_b.eval()
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            assert torch.equal(model_a(input_ids), model_b(input_ids))
