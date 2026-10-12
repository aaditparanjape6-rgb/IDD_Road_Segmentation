"""Run a trained IDD Lite checkpoint and save class masks plus overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp


CLASS_NAMES = [
    "drivable", "non_drivable", "living_things", "vehicles",
    "roadside_objects", "far_objects", "sky",
]
COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [220, 20, 60], [0, 0, 142],
    [220, 220, 0], [70, 70, 70], [70, 130, 180],
], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="IDD Lite root or its parent")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/idd_unet_best.pt"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("outputs/idd_lite_predictions"))
    args = parser.parse_args()

    root = args.data
    if not (root / "leftImg8bit").is_dir() and (root / "idd20k_lite" / "leftImg8bit").is_dir():
        root = root / "idd20k_lite"
    image_root = root / "leftImg8bit" / args.split
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image split not found: {image_root}")

    images = sorted(image_root.rglob("*_image.jpg"))
    images += sorted(image_root.rglob("*_image.png"))
    images += sorted(image_root.rglob("*_leftImg8bit.png"))
    if not images:
        raise FileNotFoundError(f"No IDD images found under {image_root}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("num_classes") != len(CLASS_NAMES):
        raise ValueError("Checkpoint is not the expected 7-class IDD Lite model.")
    width, height = checkpoint["width"], checkpoint["height"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3,
        classes=len(CLASS_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    with torch.inference_mode():
        for index, image_path in enumerate(images, start=1):
            image = Image.open(image_path).convert("RGB")
            original = np.asarray(image)
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
            array = (array - mean) / std
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
            prediction = model(tensor).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            # Resize class IDs back with nearest-neighbor before colorizing.
            label_image = Image.fromarray(prediction, mode="L").resize(
                image.size, Image.Resampling.NEAREST
            )
            labels = np.asarray(label_image, dtype=np.uint8)
            color_mask = COLORS[labels]
            overlay = (0.55 * original.astype(np.float32) + 0.45 * color_mask).clip(0, 255).astype(np.uint8)

            relative = image_path.relative_to(image_root)
            output_dir = args.output / relative.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem.replace("_image", "").replace("_leftImg8bit", "")
            Image.fromarray(labels, mode="L").save(output_dir / f"{stem}_pred.png")
            Image.fromarray(overlay, mode="RGB").save(output_dir / f"{stem}_overlay.jpg", quality=92)
            print(f"[{index}/{len(images)}] {relative}")

    print(f"Saved predicted class-ID masks and overlays under: {args.output.resolve()}")
    print("Class IDs: " + ", ".join(f"{index}={name}" for index, name in enumerate(CLASS_NAMES)))


if __name__ == "__main__":
    main()
