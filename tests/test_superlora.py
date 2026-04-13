import math
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.superlora import (
    FastfoodProjection,
    LoRAUnit,
    TuckerCore,
    KroneckerUnit,
    SuperLoRAGroup,
    SuperLoRALinear,
    inject_superlora,
    count_parameters,
)


# ── FastfoodProjection ───────────────────────────────────────────────────────

class TestFastfoodProjection:
    def test_output_shape(self):
        proj = FastfoodProjection(in_features=64, out_features=256)
        x = torch.randn(64)
        y = proj(x)
        assert y.shape == (256,)

    def test_batch_output_shape(self):
        proj = FastfoodProjection(in_features=32, out_features=128)
        x = torch.randn(4, 32)
        y = proj(x)
        assert y.shape == (4, 128)

    def test_no_trainable_params(self):
        proj = FastfoodProjection(in_features=32, out_features=128)
        trainable = [p for p in proj.parameters() if p.requires_grad]
        assert len(trainable) == 0, "Projection should have zero trainable params"

    def test_deterministic(self):
        proj = FastfoodProjection(in_features=32, out_features=128, seed=7)
        x = torch.randn(32)
        y1 = proj(x)
        y2 = proj(x)
        assert torch.allclose(y1, y2)


# ── LoRAUnit ─────────────────────────────────────────────────────────────────

class TestLoRAUnit:
    def test_output_shape(self):
        unit = LoRAUnit(d_in=32, d_out=32, rank=4)
        delta = unit()
        assert delta.shape == (32, 32)

    def test_zero_B_init(self):
        unit = LoRAUnit(d_in=64, d_out=64, rank=8)
        delta = unit()
        assert delta.abs().max().item() < 1e-6

    def test_parameter_count(self):
        d, r = 64, 8
        unit = LoRAUnit(d_in=d, d_out=d, rank=r)
        expected = 2 * d * r
        assert unit.num_parameters == expected


# ── TuckerCore ───────────────────────────────────────────────────────────────

class TestTuckerCore:
    def test_output_shape_2d(self):
        core = TuckerCore(dims=[16, 16], ranks=[4, 4])
        out = core()
        assert out.shape == (16, 16)

    def test_output_shape_3d(self):
        core = TuckerCore(dims=[8, 8, 8], ranks=[2, 2, 2])
        out = core()
        assert out.shape == (8, 8, 8)

    def test_parameter_count_2d(self):
        d, r = 16, 4
        core = TuckerCore(dims=[d, d], ranks=[r, r])
        expected = r * r + 2 * d * r   # core + 2 factors
        assert core.num_parameters == expected


# ── KroneckerUnit ────────────────────────────────────────────────────────────

class TestKroneckerUnit:
    def test_output_shape(self):
        unit = KroneckerUnit(total_dim=16, rank=4, n_splits=2)
        out = unit()
        assert out.shape == (16, 16)


# ── SuperLoRAGroup ───────────────────────────────────────────────────────────

class TestSuperLoRAGroup:
    @pytest.mark.parametrize("variant", ["lora", "lorta", "lokr"])
    def test_output_size(self, variant):
        group_size = 256
        group = SuperLoRAGroup(
            group_size=group_size,
            rank=4,
            variant=variant,
            tensor_order=3,
        )
        out = group()
        assert out.shape == (group_size,), f"Failed for variant={variant}"

    def test_projection_reduces_params(self):
        size = 1024
        g_no_proj = SuperLoRAGroup(size, rank=8, variant="lora", projection_ratio=1.0)
        g_proj    = SuperLoRAGroup(size, rank=8, variant="lora", projection_ratio=0.1)
        assert g_proj.num_parameters < g_no_proj.num_parameters


# ── SuperLoRALinear ──────────────────────────────────────────────────────────

class TestSuperLoRALinear:
    def test_forward_shape(self):
        linear = nn.Linear(64, 64)
        adapted = SuperLoRALinear(linear, rank=4, variant="lora")
        x = torch.randn(2, 64)
        y = adapted(x)
        assert y.shape == (2, 64)

    def test_base_frozen(self):
        linear = nn.Linear(64, 64)
        adapted = SuperLoRALinear(linear, rank=4)
        for p in adapted.base_layer.parameters():
            assert not p.requires_grad

    def test_adapter_trainable(self):
        linear = nn.Linear(32, 32)
        adapted = SuperLoRALinear(linear, rank=4)
        trainable = [p for p in adapted.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_adapter_params_lt_full(self):
        """Adapter should use far fewer params than d^2."""
        linear = nn.Linear(768, 768)
        adapted = SuperLoRALinear(linear, rank=8)
        full_params = 768 * 768
        assert adapted.num_adapter_parameters < full_params / 10


# ── inject_superlora ─────────────────────────────────────────────────────────

class TestInjectSuperLoRA:
    def _simple_vit_like(self):
        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(64, 64)
                self.value = nn.Linear(64, 64)
                self.out   = nn.Linear(64, 64)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = FakeAttention()
                self.head = nn.Linear(64, 10)

        return FakeModel()

    def test_layers_replaced(self):
        model = self._simple_vit_like()
        model = inject_superlora(model, target_modules=("query", "value"), rank=4)
        assert isinstance(model.attn.query, SuperLoRALinear)
        assert isinstance(model.attn.value, SuperLoRALinear)
        # 'out' and 'head' should NOT be replaced
        assert isinstance(model.attn.out, nn.Linear)
        assert isinstance(model.head,     nn.Linear)

    def test_parameter_reduction(self):
        model = self._simple_vit_like()
        full_params = count_parameters(model)["total"]
        model = inject_superlora(model, target_modules=("query", "value"), rank=4)
        trainable = count_parameters(model)["trainable"]
        # Head (64×10) + adapter params must be fewer than all q/v weights
        assert trainable < full_params

    def test_forward_still_works(self):
        model = self._simple_vit_like()
        model = inject_superlora(model, target_modules=("query", "value"), rank=4)
        x = torch.randn(2, 64)
        # Manually call attn to verify no errors
        _ = model.attn.query(x)
        _ = model.attn.value(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])