"""Train a multi-class semantic segmentation U-Net on IDD Lite or full IDD.

Expected IDD layout (after running AutoNUE's createLabels.py):
Expected IDD Lite layout after extracting its official archive:

    IDD/
      leftImg8bit/train/<drive>/*_leftImg8bit.png
      leftImg8bit/val/<drive>/*_leftImg8bit.png
      gtFine/train/<drive>/*_gtFine_labellevel1Ids.png  # IDD Lite, default
      gtFine/val/<drive>/*_gtFine_labellevel1Ids.png
      leftImg8bit/train/<drive>/*_image.jpg
      leftImg8bit/val/<drive>/*_image.jpg
      gtFine/train/<drive>/*_label.png
      gtFine/val/<drive>/*_label.png

IDD Lite uses seven Level-1 classes (0..6); full IDD can use Level-3 IDs via
--label-level level3 (0..26). In either case, label 255 is ignored.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp


NUM_CLASSES = 7  # IDD Lite Level-1 IDs 0..6
IGNORE_ID = 255
CLASS_NAMES = [
    "drivable", "non_drivable", "living_things", "vehicles",
    "roadside_objects", "far_objects", "sky",
]


def paired_files(root: Path, split: str, label_level: str) -> list[tuple[Path, Path]]:
    if not (root / "leftImg8bit").is_dir() and (root / "idd20k_lite" / "leftImg8bit").is_dir():
        root = root / "idd20k_lite"
    image_root = root / "leftImg8bit" / split
    label_root = root / "gtFine" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f"Expected image and label directories for '{split}':\n"
            f"  {image_root}\n  {label_root}\n"
            "Check the extracted IDD Lite directory layout; see README_IDD.md."
        )

    images = sorted(image_root.rglob("*_leftImg8bit.png"))
    images += sorted(image_root.rglob("*_image.jpg"))
    images += sorted(image_root.rglob("*_image.png"))
    if not images:
        images = sorted(path for path in image_root.rglob("*")
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    pairs: list[tuple[Path, Path]] = []
    for image_path in images:
        relative = image_path.relative_to(image_root)
        stem = image_path.stem
        for suffix in ("_leftImg8bit", "_image"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        candidates = [
            label_root / relative.parent / f"{stem}_label.png",
            label_root / relative.parent / f"{stem}_gtFine_label{label_level}Ids.png",
        ]
        label_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if label_path is not None:
            pairs.append((image_path, label_path))

    if not pairs:
        raise FileNotFoundError(
            f"No paired image/mask files found under {image_root} and {label_root}. "
            "Check the IDD directory layout and label suffix."
        )
    return pairs


class IDDDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], width: int, height: int,
                 augment: bool = False):
        self.pairs = pairs
        self.width = width
        self.height = height
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB").resize(
            (self.width, self.height), Image.Resampling.BILINEAR
        )
        label = Image.open(label_path).convert("L").resize(
            (self.width, self.height), Image.Resampling.NEAREST
        )

        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                label = ImageOps.mirror(label)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        # ImageNet normalization matches the pretrained ResNet encoder.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        label_tensor = torch.from_numpy(np.asarray(label, dtype=np.int64).copy())
        return image_tensor, label_tensor


def compute_class_weights(pairs: list[tuple[Path, Path]]) -> torch.Tensor:
    """Use mild inverse-square-root frequency weights to help rare classes."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, label_path in pairs:
        labels = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
        valid = labels < NUM_CLASSES
        counts += np.bincount(labels[valid], minlength=NUM_CLASSES)[:NUM_CLASSES]
    present = counts > 0
    if not present.any():
        raise ValueError("No valid class IDs found in training masks (expected IDs 0..6).")
    frequencies = counts[present] / counts[present].sum()
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    weights[present] = 1.0 / np.sqrt(frequencies)
    weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)


def update_confusion(confusion: torch.Tensor, prediction: torch.Tensor,
                     target: torch.Tensor) -> None:
    valid = target != IGNORE_ID
    target = target[valid]
    prediction = prediction[valid]
    in_range = (target >= 0) & (target < NUM_CLASSES)
    encoded = NUM_CLASSES * target[in_range] + prediction[in_range]
    confusion += torch.bincount(
        encoded, minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def mean_iou(confusion: torch.Tensor) -> tuple[float, list[float]]:
    intersection = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    ious = torch.where(union > 0, intersection / union, torch.nan)
    return torch.nanmean(ious).item(), ious.tolist()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: torch.optim.Optimizer | None) -> tuple[float, float, list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_pixels = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        pixels = int((targets != IGNORE_ID).sum().item())
        total_loss += loss.item() * pixels
        total_pixels += pixels
        update_confusion(confusion, logits.argmax(dim=1), targets)

    score, per_class = mean_iou(confusion)
    return total_loss / max(total_pixels, 1), score, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="IDD dataset root")
    parser.add_argument("--output", type=Path, default=Path("outputs/idd_unet_best.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--label-level", choices=("level1", "level3"), default="level1",
                        help="Use level1 for IDD Lite (7 classes), level3 for full IDD (27 classes).")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--non-drivable-boost", type=float, default=1.5,
                        help="Extra loss multiplier for IDD Lite class 1")
    parser.add_argument("--roadside-boost", type=float, default=2.0,
                        help="Extra loss multiplier for IDD Lite class 4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    global NUM_CLASSES, CLASS_NAMES
    if args.label_level == "level3":
        NUM_CLASSES = 27
        CLASS_NAMES = [
            "road", "parking_or_drivable_fallback", "sidewalk",
            "rail_track_or_non_drivable_fallback", "person_or_animal", "rider",
            "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus",
            "large_vehicle_fallback", "curb", "wall", "fence", "guard_rail",
            "billboard", "traffic_sign", "traffic_light", "pole_or_polegroup",
            "obstacle_fallback", "building", "bridge_or_tunnel", "vegetation", "sky",
            "miscellaneous",
        ]
        if args.width == 256 and args.height == 128:
            args.width, args.height = 512, 288

    train_pairs = paired_files(args.data, "train", args.label_level)
    val_pairs = paired_files(args.data, "val", args.label_level)
    train_loader = DataLoader(
        IDDDataset(train_pairs, args.width, args.height, augment=True), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        IDDDataset(val_pairs, args.width, args.height), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet",
        in_channels=3, classes=NUM_CLASSES,
    ).to(device)
    class_weights = compute_class_weights(train_pairs)
    if args.label_level == "level1":
        class_weights[1] *= args.non_drivable_boost
        class_weights[4] *= args.roadside_boost
        class_weights /= class_weights.mean()
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}; train: {len(train_pairs)}; val: {len(val_pairs)}")
    print("Training class weights:", [round(float(weight), 3) for weight in class_weights.cpu()])
    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_miou, val_class_ious = run_epoch(model, val_loader, criterion, device, None)
        class_scores = " ".join(
            f"{name}={score:.3f}" for name, score in zip(CLASS_NAMES, val_class_ious)
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train loss={train_loss:.4f} mIoU={train_miou:.4f} | "
            f"val loss={val_loss:.4f} mIoU={val_miou:.4f} | class IoU: {class_scores}"
        )
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "model": model.state_dict(), "num_classes": NUM_CLASSES,
                "ignore_id": IGNORE_ID, "width": args.width, "height": args.height,
                "class_names": CLASS_NAMES, "val_miou": best_miou,
                "val_class_ious": val_class_ious,
            }, args.output)
            print(f"  saved best checkpoint: {args.output}")

    print(f"Best validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
