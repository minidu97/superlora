import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


# ── ImageNet statistics (used because ViT is pretrained on ImageNet) ─────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_dataloaders(
    dataset: str = "cifar100",
    data_root: str = "./data",
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:

    dataset = dataset.lower()

    train_tf = get_transforms(image_size, train=True)
    val_tf   = get_transforms(image_size, train=False)

    if dataset == "cifar100":
        DatasetClass = datasets.CIFAR100
    elif dataset == "cifar10":
        DatasetClass = datasets.CIFAR10
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose 'cifar10' or 'cifar100'.")

    train_ds = DatasetClass(data_root, train=True,  download=True, transform=train_tf)
    val_ds   = DatasetClass(data_root, train=False, download=True, transform=val_tf)

    # pin_memory only works on CUDA — disable on CPU/MPS to avoid warnings
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    print(
        f"[Data] {dataset.upper()}  "
        f"train={len(train_ds):,}  val={len(val_ds):,}  "
        f"batch={batch_size}  image={image_size}x{image_size}"
    )
    return train_loader, val_loader