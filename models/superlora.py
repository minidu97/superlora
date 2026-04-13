import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


#Fastfood Projection (fixed, no learnable params)

class FastfoodProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Pad to power of 2 for fast Hadamard transform
        self.padded_size = 2 ** math.ceil(math.log2(out_features))

        rng = np.random.RandomState(seed)

        # Random vectors (fixed — registered as buffers, not parameters)
        G = torch.tensor(rng.randn(self.padded_size), dtype=torch.float32)
        B = torch.tensor(rng.choice([-1, 1], size=self.padded_size), dtype=torch.float32)
        perm = torch.tensor(rng.permutation(self.padded_size), dtype=torch.long)

        self.register_buffer("G", G)
        self.register_buffer("B", B)
        self.register_buffer("perm", perm)

    def _hadamard_transform(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        h = x.clone()
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                a = h[..., i:i + step]
                b = h[..., i + step:i + step * 2]
                h[..., i:i + step] = a + b
                h[..., i + step:i + step * 2] = a - b
            step *= 2
        return h / math.sqrt(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features)
        batch = flat.shape[0]

        # Pad to padded_size
        pad_len = self.padded_size - self.in_features
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        # Step 1: diagonal B
        out = flat * self.B.unsqueeze(0)
        # Step 2: Hadamard H
        out = self._hadamard_transform(out)
        # Step 3: permutation Π
        out = out[:, self.perm]
        # Step 4: diagonal G
        out = out * self.G.unsqueeze(0)
        # Step 5: truncated Hadamard H'
        out = self._hadamard_transform(out)
        # Truncate to out_features
        out = out[:, :self.out_features]

        if x.dim() == 1:
            return out.squeeze(0)
        return out.reshape(*x.shape[:-1], self.out_features)

#Tucker-Decomposed Core (LoRTA building block)

class TuckerCore(nn.Module):
    def __init__(self, dims: List[int], ranks: List[int]):
        super().__init__()
        assert len(dims) == len(ranks), "dims and ranks must have same length"
        self.dims = dims
        self.ranks = ranks
        self.order = len(dims)

        # Core tensor
        self.core = nn.Parameter(torch.empty(*ranks))
        # Mode factors
        self.factors = nn.ParameterList([
            nn.Parameter(torch.empty(d, r))
            for d, r in zip(dims, ranks)
        ])
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.core, std=0.02)
        for f in self.factors:
            nn.init.orthogonal_(f)

    def forward(self) -> torch.Tensor:
        result = self.core  # shape: (r0, r1, ..., r_{M-1})
        for m, A in enumerate(self.factors):
            # Move mode m to front so we can contract along dim-0
            result = result.movedim(m, 0)                      # (r_m, ...)
            # Contract: (r_m, ...) x (d_m, r_m)^T => (d_m, ...)
            shape_rest = result.shape[1:]
            result = result.reshape(result.shape[0], -1)       # (r_m, rest)
            result = A @ result                                 # (d_m, rest)
            result = result.reshape(A.shape[0], *shape_rest)   # (d_m, ...)
            # Move d_m back to position m
            result = result.movedim(0, m)                      # (..., d_m, ...)
        return result

    @property
    def num_parameters(self) -> int:
        n = self.core.numel()
        for f in self.factors:
            n += f.numel()
        return n

#Standard 2-D LoRA unit  (baseline)

class LoRAUnit(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scaling

    @property
    def num_parameters(self) -> int:
        return self.A.numel() + self.B.numel()

#Kronecker-product unit  (LoKr / LoNKr)

class KroneckerUnit(nn.Module):
    def __init__(self, total_dim: int, rank: int, n_splits: int = 2):
        super().__init__()
        self.n_splits = n_splits
        # Split dimension evenly
        split_dim = max(1, round(total_dim ** (1 / n_splits)))
        self.units = nn.ModuleList([
            LoRAUnit(split_dim, split_dim, rank)
            for _ in range(n_splits)
        ])
        self.total_dim = total_dim

    def forward(self) -> torch.Tensor:
        result = self.units[0]()
        for unit in self.units[1:]:
            result = torch.kron(result, unit())
        # Trim/pad to total_dim × total_dim
        d = self.total_dim
        result = result[:d, :d]
        return result

    @property
    def num_parameters(self) -> int:
        return sum(u.num_parameters for u in self.units)

#SuperLoRA Group

class SuperLoRAGroup(nn.Module):
    def __init__(
        self,
        group_size: int,
        rank: int,
        variant: str = "lora",
        tensor_order: int = 2,
        n_splits: int = 2,
        projection_ratio: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.group_size = group_size
        self.variant = variant.lower()
        self.use_projection = projection_ratio < 1.0
        self.projection_ratio = projection_ratio

        #inner (pre-projection) size
        inner_size = max(1, int(group_size * projection_ratio))

        if self.variant == "lora":
            d = int(math.sqrt(inner_size))
            self.core_unit = LoRAUnit(d, d, rank)
        elif self.variant == "lorta":
            d = max(2, int(inner_size ** (1 / tensor_order)))
            dims = [d] * tensor_order
            ranks = [rank] * tensor_order
            self.core_unit = TuckerCore(dims, ranks)
        elif self.variant in ("lokr", "lonkr"):
            self.core_unit = KroneckerUnit(
                int(math.sqrt(inner_size)), rank, n_splits=n_splits
            )
        else:
            raise ValueError(f"Unknown variant '{variant}'")

        #optional fixed projection
        if self.use_projection:
            self.projection = FastfoodProjection(inner_size, group_size, seed=seed)
        else:
            self.projection = None

    def forward(self) -> torch.Tensor:
        delta = self.core_unit()
        delta = delta.reshape(-1)

        if self.projection is not None:
            delta = self.projection(delta)

        # Pad or trim to exact group_size
        if delta.numel() < self.group_size:
            delta = F.pad(delta, (0, self.group_size - delta.numel()))
        else:
            delta = delta[:self.group_size]

        return delta

    @property
    def num_parameters(self) -> int:
        return self.core_unit.num_parameters

#SuperLoRA Adapter  (wraps a single nn.Linear layer)

class SuperLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        variant: str = "lora",
        n_groups: int = 1,
        tensor_order: int = 3,
        n_splits: int = 2,
        projection_ratio: float = 1.0,
        alpha: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.d_out, self.d_in = base_layer.weight.shape
        total = self.d_out * self.d_in
        self.n_groups = n_groups

        # Freeze base weights
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        # Split total weight elements into G groups
        sizes = self._compute_group_sizes(total, n_groups)
        self.groups = nn.ModuleList([
            SuperLoRAGroup(
                group_size=s,
                rank=rank,
                variant=variant,
                tensor_order=tensor_order,
                n_splits=n_splits,
                projection_ratio=projection_ratio,
                seed=seed + i,
            )
            for i, s in enumerate(sizes)
        ])

    @staticmethod
    def _compute_group_sizes(total: int, n_groups: int) -> List[int]:
        base, remainder = divmod(total, n_groups)
        return [base + (1 if i < remainder else 0) for i in range(n_groups)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct ∆W from all groups
        delta_flat = torch.cat([g() for g in self.groups])
        delta_W = delta_flat[:self.d_out * self.d_in].reshape(self.d_out, self.d_in)
        # Forward: (W + ∆W) x
        return F.linear(x, self.base_layer.weight + delta_W, self.base_layer.bias)

    @property
    def num_adapter_parameters(self) -> int:
        return sum(g.num_parameters for g in self.groups)

    def __repr__(self) -> str:
        return (
            f"SuperLoRALinear(d_in={self.d_in}, d_out={self.d_out}, "
            f"groups={self.n_groups}, adapter_params={self.num_adapter_parameters:,})"
        )

#Model-level helper: inject SuperLoRA into attention Q/V layers

def inject_superlora(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("query", "value"),
    rank: int = 8,
    variant: str = "lora",
    n_groups: int = 1,
    tensor_order: int = 3,
    n_splits: int = 2,
    projection_ratio: float = 1.0,
    alpha: float = 1.0,
) -> nn.Module:

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if parent_name == "" else dict(model.named_modules())[parent_name]

        adapted = SuperLoRALinear(
            base_layer=module,
            rank=rank,
            variant=variant,
            n_groups=n_groups,
            tensor_order=tensor_order,
            n_splits=n_splits,
            projection_ratio=projection_ratio,
            alpha=alpha,
        )
        setattr(parent, child_name, adapted)
        replaced += 1

    print(f"[SuperLoRA] Replaced {replaced} linear layer(s).")
    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}