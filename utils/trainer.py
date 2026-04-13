import os
import time
import math
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR

from configs.config import TrainConfig


#Device helper
def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


#Metrics
class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append((correct_k / batch_size).item())
        return res


#Build optimizer
def build_optimizer(model: nn.Module, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay, nesterov=True)
    elif cfg.optimizer == "adamw":
        return AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total_steps = cfg.num_steps
    if cfg.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
        )
    elif cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    elif cfg.scheduler == "step":
        return StepLR(optimizer, step_size=total_steps // 3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")


#Evaluation
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")
    criterion  = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        top1,  = accuracy(logits, labels, topk=(1,))
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(top1, images.size(0))

    return {"val_loss": loss_meter.avg, "val_acc": acc1_meter.avg}


#Training loop
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> Dict[str, list]:
    device    = resolve_device(cfg.device)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    save_dir = Path(cfg.save_dir) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"step": [], "train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    step = 0
    t0 = time.time()

    train_iter = iter(train_loader)

    while step < cfg.num_steps:
        model.train()

        #Reload iterator when exhausted
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        #Gradient clipping for stability
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()
        scheduler.step()
        step += 1

        top1, = accuracy(logits, labels, topk=(1,))

        #Logging
        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]
            print(
                f"  step {step:5d}/{cfg.num_steps}  "
                f"loss={loss.item():.4f}  acc={top1:.3f}  "
                f"lr={lr_now:.5f}  t={elapsed:.0f}s"
            )
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["train_acc"].append(top1)
            history["lr"].append(lr_now)

        #Validation
        if step % cfg.eval_interval == 0 or step == cfg.num_steps:
            val_metrics = evaluate(model, val_loader, device)
            val_acc = val_metrics["val_acc"]
            history["val_acc"].append(val_acc)
            print(
                f"  [eval] step={step}  val_acc={val_acc:.4f}  "
                f"val_loss={val_metrics['val_loss']:.4f}"
            )

            #Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = save_dir / "best_model.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"  ✓ best checkpoint saved  val_acc={val_acc:.4f}")

    #Save history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Done.  Best val_acc={best_val_acc:.4f}  "
          f"Total time={(time.time()-t0)/60:.1f} min")
    return history