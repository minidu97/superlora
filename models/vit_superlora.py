import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from models.superlora import inject_superlora, count_parameters


class SuperLoRAViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224",
        # SuperLoRA hyper-parameters
        rank: int = 8,
        variant: str = "lora",
        n_groups: int = 1,
        tensor_order: int = 3,
        n_splits: int = 2,
        projection_ratio: float = 1.0,
        alpha: float = 1.0,
        target_modules: Tuple[str, ...] = ("query", "value", "qkv"),
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required: pip install timm"
            )

        # ── Load backbone ────────────────────────────────────────────────
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove classification head
        )

        # ── Freeze all backbone parameters ───────────────────────────────
        for p in self.vit.parameters():
            p.requires_grad_(False)

        # ── Inject SuperLoRA adapters ────────────────────────────────────
        self.vit = inject_superlora(
            self.vit,
            target_modules=target_modules,
            rank=rank,
            variant=variant,
            n_groups=n_groups,
            tensor_order=tensor_order,
            n_splits=n_splits,
            projection_ratio=projection_ratio,
            alpha=alpha,
        )

        # ── Classification head (fully trainable) ────────────────────────
        embed_dim = self.vit.num_features
        self.head = nn.Linear(embed_dim, num_classes)

        # ── Log parameter counts ─────────────────────────────────────────
        stats = count_parameters(self)
        print(
            f"[SuperLoRAViT] backbone={model_name}  variant={variant}  "
            f"rank={rank}  groups={n_groups}\n"
            f"  trainable : {stats['trainable']:>12,}\n"
            f"  frozen    : {stats['frozen']:>12,}\n"
            f"  total     : {stats['total']:>12,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit(x)
        return self.head(features)

    def get_adapter_params(self):
        return [p for p in self.parameters() if p.requires_grad]