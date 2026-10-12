"""Turn IDD Lite still frames into a simple slideshow video for demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Folder containing IDD Lite *_image.jpg frames")
    parser.add_argument("--output", type=Path, default=Path("outputs/idd_lite_slideshow.mp4"))
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Playback frame rate; 1 fps shows each still for about a second")
    args = parser.parse_args()

    frames = sorted(args.input_dir.glob("*_image.jpg"))
    if not frames:
        raise FileNotFoundError(f"No *_image.jpg frames found in {args.input_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise OSError(f"Could not read image: {frames[0]}")
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Could not create MP4: {args.output}")

    written = 0
    try:
        for path in frames:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
            written += 1
    finally:
        writer.release()

    if written == 0:
        raise RuntimeError("No readable images were written")
    print(f"Created slideshow from {written} still images: {args.output}")
    print("This is a slideshow demo, not continuous driving footage.")


if __name__ == "__main__":
    main()
