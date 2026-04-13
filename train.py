import argparse
import random
import numpy as np
import torch
import sys, os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import PRESETS, TrainConfig, SuperLoRAConfig
from models.vit_superlora import SuperLoRAViT
from utils.data import get_dataloaders
from utils.trainer import train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="SuperLoRA Training")

    # Preset config
    p.add_argument("--preset", type=str, default="superlora_2d",
                   choices=list(PRESETS.keys()),
                   help="Named experiment preset from configs/config.py")

    # Dataset
    p.add_argument("--dataset",    type=str,   default=None,
                   choices=["cifar10", "cifar100"])
    p.add_argument("--data_root",  type=str,   default="./data")
    p.add_argument("--num_classes",type=int,   default=None)

    # Training
    p.add_argument("--steps",      type=int,   default=None, dest="num_steps")
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None, dest="learning_rate")
    p.add_argument("--optimizer",  type=str,   default=None, choices=["sgd", "adamw"])
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--save_dir",   type=str,   default=None)
    p.add_argument("--run_name",   type=str,   default=None)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--num_workers",type=int,   default=None)

    # SuperLoRA hyper-parameters (override preset)
    p.add_argument("--rank",              type=int,   default=None)
    p.add_argument("--variant",           type=str,   default=None,
                   choices=["lora", "lorta", "lokr", "lonkr"])
    p.add_argument("--n_groups",          type=int,   default=None)
    p.add_argument("--tensor_order",      type=int,   default=None)
    p.add_argument("--n_splits",          type=int,   default=None)
    p.add_argument("--projection_ratio",  type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # ── Load preset ──────────────────────────────────────────────────────────
    cfg: TrainConfig = PRESETS[args.preset]()

    # ── Apply CLI overrides ──────────────────────────────────────────────────
    for key in ("dataset", "data_root", "num_classes", "num_steps",
                "batch_size", "learning_rate", "optimizer",
                "device", "save_dir", "run_name", "seed", "num_workers"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)

    for key in ("rank", "variant", "n_groups", "tensor_order",
                "n_splits", "projection_ratio"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg.superlora, key, val)

    set_seed(cfg.seed)

    print("=" * 60)
    print(f"  SuperLoRA — preset: {args.preset}")
    print(f"  dataset  : {cfg.dataset}  classes={cfg.num_classes}")
    print(f"  variant  : {cfg.superlora.variant}  rank={cfg.superlora.rank}")
    print(f"  groups   : {cfg.superlora.n_groups}  proj_ratio={cfg.superlora.projection_ratio}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        dataset    = cfg.dataset,
        data_root  = cfg.data_root,
        image_size = cfg.image_size,
        batch_size = cfg.batch_size,
        num_workers= cfg.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = SuperLoRAViT(
        num_classes      = cfg.num_classes,
        pretrained       = cfg.pretrained,
        model_name       = cfg.model_name,
        rank             = cfg.superlora.rank,
        variant          = cfg.superlora.variant,
        n_groups         = cfg.superlora.n_groups,
        tensor_order     = cfg.superlora.tensor_order,
        n_splits         = cfg.superlora.n_splits,
        projection_ratio = cfg.superlora.projection_ratio,
        alpha            = cfg.superlora.alpha,
        target_modules   = cfg.superlora.target_modules,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    history = train(model, train_loader, val_loader, cfg)


if __name__ == "__main__":
    main()