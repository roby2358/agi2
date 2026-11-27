"""Tests for src/model.py."""

import pytest
import torch
from src.model import HegelModel


def _copy_model() -> HegelModel:
    """Two slots of width 3: head 0 copies the previous token into slot 1."""
    model = HegelModel(
        vocab_size=3, d_model=6, n_heads=1, d_head=3, d_ff=1, max_offset=1
    )
    model.embed[:, :3] = torch.eye(3)
    model.rel_bias[0] = torch.tensor([-30.0, 30.0])
    model.w_v[0, :, :3] = torch.eye(3)
    model.w_o[0, 3:, :] = torch.eye(3)
    return model


@pytest.mark.unit
class TestHegelModel:
    def test_starts_at_zero_and_is_never_trainable(self) -> None:
        model = HegelModel(5, 8, 2, 4, 7, 2)
        assert model.nonzero_count() == 0
        assert not any(p.requires_grad for p in model.parameters())

    def test_zero_model_is_uniform(self) -> None:
        model = HegelModel(5, 8, 2, 4, 7, 2)
        out = model(torch.tensor([[0, 3, 1]]))
        assert out.shape == (1, 3, 5)
        assert torch.allclose(out.exp(), torch.full((1, 3, 5), 0.2))

    def test_head_copies_the_previous_token(self) -> None:
        model = _copy_model()
        x = model.embed[torch.tensor([[2, 0, 1]])]
        written = model.attend(x)[0]
        assert torch.allclose(written[1], torch.tensor([0, 0, 0, 0, 0, 1.0]))
        assert torch.allclose(written[2], torch.tensor([0, 0, 0, 1.0, 0, 0]))

    def test_head_writes_nothing_without_a_predecessor(self) -> None:
        model = _copy_model()
        x = model.embed[torch.tensor([[2, 0, 1]])]
        assert torch.all(model.attend(x)[0, 0].abs() < 1e-9)

    def test_no_position_sees_the_future(self) -> None:
        model = _copy_model()
        model.unembed[:, 3:] = torch.eye(3)
        a = model(torch.tensor([[2, 0, 1]]))
        b = model(torch.tensor([[2, 0, 2]]))
        assert torch.equal(a[0, :2], b[0, :2])

    def test_parameter_count(self) -> None:
        model = HegelModel(5, 8, 2, 4, 7, 2)
        expected = 5 * 8 + 2 * 3 + 2 * 4 * 8 + 2 * 8 * 4 + 7 * 8 + 7 + 8 * 7 + 8 + 5 * 8
        assert model.parameter_count() == expected
