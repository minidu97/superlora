from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class SuperLoRAConfig:
    # Factorization rank r
    rank: int = 8

    # Variant: 'lora' | 'lorta' | 'lokr' | 'lonkr'
    variant: str = "lora"

    # Number of groups G  (1 = group-wise across all layers)
    n_groups: int = 1

    # Tensor order M  (2 = LoRA, 3/4/5 = LoRTA)
    tensor_order: int = 2

    # Kronecker splits K  (2 = LoKr, >2 = LoNKr)
    n_splits: int = 2

    # Projection ratio ρ  (1.0 = no projection)
    projection_ratio: float = 1.0

    # LoRA scaling alpha
    alpha: float = 1.0

    # Which sublayer names to adapt
    target_modules: Tuple[str, ...] = ("query", "value", "qkv")


@dataclass
class TrainConfig:
    # Dataset
    dataset: str = "cifar100"        # 'cifar100' | 'cifar10'
    data_root: str = "./data"
    num_classes: int = 100
    image_size: int = 224

    # Backbone
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True

    # Training schedule (matches paper: 5000 steps)
    num_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "sgd"           # 'sgd' | 'adamw'

    # Learning-rate scheduler
    scheduler: str = "onecycle"      # 'onecycle' | 'cosine' | 'step'
    max_lr: float = 0.05

    # Misc
    seed: int = 42
    device: str = "auto"             # 'auto' | 'cpu' | 'cuda' | 'mps'
    num_workers: int = 4
    log_interval: int = 50
    eval_interval: int = 200
    save_dir: str = "./checkpoints"
    run_name: str = "superlora_run"

    # SuperLoRA config (nested)
    superlora: SuperLoRAConfig = field(default_factory=SuperLoRAConfig)


#Preset configs from the paper

def lora_baseline() -> TrainConfig:
    cfg = TrainConfig()
    cfg.superlora = SuperLoRAConfig(rank=8, variant="lora", n_groups=24)
    cfg.run_name = "lora_baseline"
    return cfg


def superlora_2d() -> TrainConfig:
    cfg = TrainConfig()
    cfg.superlora = SuperLoRAConfig(rank=8, variant="lora", n_groups=1)
    cfg.run_name = "superlora_2d"
    return cfg


def superlora_2d_reshape() -> TrainConfig:
    cfg = TrainConfig()
    cfg.superlora = SuperLoRAConfig(rank=16, variant="lora", n_groups=1)
    cfg.run_name = "superlora_2d_reshape"
    return cfg


def lorta_3d() -> TrainConfig:
    cfg = TrainConfig()
    cfg.superlora = SuperLoRAConfig(
        rank=4, variant="lorta", n_groups=1, tensor_order=3
    )
    cfg.run_name = "lorta_3d"
    return cfg


def lorta_4d() -> TrainConfig:
    cfg = TrainConfig()
    cfg.superlora = SuperLoRAConfig(
        rank=4, variant="lorta", n_groups=1, tensor_order=4
    )
    cfg.run_name = "lorta_4d"
    return cfg


def superlora_projected() -> TrainConfig:
    cfg = TrainConfig()
    cfg.superlora = SuperLoRAConfig(
        rank=8, variant="lora", n_groups=1, projection_ratio=0.1
    )
    cfg.run_name = "superlora_projected"
    return cfg


def lokr() -> TrainConfig:
    cfg = TrainConfig()
    cfg.superlora = SuperLoRAConfig(
        rank=8, variant="lokr", n_groups=1, n_splits=2
    )
    cfg.run_name = "lokr"
    return cfg


PRESETS = {
    "lora_baseline": lora_baseline,
    "superlora_2d": superlora_2d,
    "superlora_2d_reshape": superlora_2d_reshape,
    "lorta_3d": lorta_3d,
    "lorta_4d": lorta_4d,
    "superlora_projected": superlora_projected,
    "lokr": lokr,
}