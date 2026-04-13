# SuperLoRA — Parameter-Efficient Unified Adaptation for Large Vision Models

Implementation of **SuperLoRA** from:
> Chen et al., *"SuperLoRA: Parameter-Efficient Unified Adaptation for Large Vision Models"*, CVPRW 2024.

---

## Project Structure

```
superlora/
├── models/
│   ├── superlora.py          # Core: FastfoodProjection, LoRAUnit, TuckerCore,
│   │                         #       KroneckerUnit, SuperLoRAGroup, SuperLoRALinear,
│   │                         #       inject_superlora()
│   └── vit_superlora.py      # SuperLoRAViT — ViT-Base + adapters
├── configs/
│   └── config.py             # SuperLoRAConfig, TrainConfig, and paper presets
├── utils/
│   ├── data.py               # CIFAR-10 / CIFAR-100 data loaders
│   └── trainer.py            # Training loop, evaluation, checkpointing
├── experiments/
│   └── pareto_analysis.py    # Reproduce accuracy vs. #params Pareto curve
├── tests/
│   └── test_superlora.py     # Unit tests (pytest)
├── train.py                  # Main training entry point
├── requirements.txt
└── .vscode/
    ├── launch.json           # Ready-to-run VS Code launch configs
    └── settings.json
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Run unit tests

```bash
python -m pytest tests/ -v
```

### 3. Smoke test (CPU, no GPU needed)

```bash
python train.py --preset lora_baseline --steps 10 --batch_size 8 --device cpu
```

### 4. Full training (reproduces paper experiments)

| Preset | Description |
|--------|-------------|
| `lora_baseline` | Standard LoRA, weight-wise grouping (paper baseline) |
| `superlora_2d` | Group-wise SuperLoRA 2-D |
| `superlora_2d_reshape` | Group-wise SuperLoRA 2-D with reshape |
| `lorta_3d` | **LoRTA 3-D** — 10× param reduction vs. LoRA |
| `lorta_4d` | LoRTA 4-D |
| `superlora_projected` | SuperLoRA + fastfood projection ρ=0.1 |
| `lokr` | LoKr (Kronecker, K=2) |

```bash
python train.py --preset lorta_3d
python train.py --preset superlora_2d --rank 16
python train.py --preset lora_baseline --dataset cifar10 --num_classes 10
```

### 5. Run in VS Code

Open the project folder in VS Code and press **F5** (or use the Run & Debug panel).
Seven pre-configured launch profiles are available in `.vscode/launch.json`.

---

## Key Classes

### `SuperLoRALinear`
Wraps a frozen `nn.Linear` with a SuperLoRA adapter:
```python
from models.superlora import SuperLoRALinear
import torch.nn as nn

layer   = nn.Linear(768, 768)
adapted = SuperLoRALinear(layer, rank=8, variant="lorta", tensor_order=3)
# Only adapter params are trainable; base layer is frozen
print(adapted.num_adapter_parameters)
```

### `inject_superlora`
Inject adapters into an existing model:
```python
from models.superlora import inject_superlora
model = inject_superlora(
    model,
    target_modules=("query", "value"),   # substring match
    rank=8,
    variant="lorta",
    n_groups=1,
    tensor_order=3,
    projection_ratio=0.1,                # enable fastfood projection
)
```

### `SuperLoRAConfig` / `TrainConfig`
```python
from configs.config import TrainConfig, SuperLoRAConfig

cfg = TrainConfig(
    dataset="cifar100",
    num_classes=100,
    num_steps=5000,
    superlora=SuperLoRAConfig(rank=4, variant="lorta", tensor_order=3),
)
```

---

## Variants Implemented

| Variant flag | Paper name | Description |
|---|---|---|
| `lora` | LoRA | Standard 2-D low-rank adapter |
| `lorta` | LoRTA | High-order Tucker decomposition |
| `lokr` | LoKr | Kronecker product (K=2 splits) |
| `lonkr` | LoNKr | N-split Kronecker (K>2) |

All variants support optional **fastfood projection** (`projection_ratio < 1`)
and **group-wise sharing** (`n_groups`).

---

## Results (from paper)

Transfer learning from ImageNet21k → CIFAR-100 (ViT-Base):

| Method | # Params | Accuracy |
|--------|----------|---------|
| LoRA (baseline) | ~4×10⁴ | ~0.90 |
| SuperLoRA 2D | ~1×10⁴ | ~0.90 (+3–4× efficient) |
| **LoRTA 3D** | **~4×10³** | **~0.91 (+10× efficient)** |



for running as a recomended way it goes with 
python run_all.py --only lora_baseline superlora_2d lorta_3d --steps 5000