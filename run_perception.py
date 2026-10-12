"""Run the IDD Lite segmentation model on an image or video.

Produces a class-color overlay and a simple centerline proposal derived from
the pixels predicted as IDD Lite's drivable class (class ID 0). The centerline
is a geometric heuristic for demonstration; it is not a vehicle controller.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


CLASS_NAMES = [
    "drivable", "non-drivable", "living things", "vehicles",
    "roadside objects", "far objects", "sky",
]
# High-contrast, color-blind-friendly palette (RGB).
CLASS_COLORS_RGB = np.array([
    [0, 191, 196], [248, 118, 109], [124, 174, 0], [199, 124, 255],
    [230, 156, 0], [97, 156, 255], [245, 100, 227],
], dtype=np.uint8)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("num_classes") != len(CLASS_NAMES):
        raise ValueError("Expected a 7-class IDD Lite checkpoint.")
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3,
        classes=len(CLASS_NAMES),
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint["width"]), int(checkpoint["height"])


def predict_mask(frame_bgr: np.ndarray, model, width: int, height: int,
                 device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0)
    if device.type == "cuda":
        tensor = tensor.to(device=device, memory_format=torch.channels_last)
    else:
        tensor = tensor.to(device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(tensor)
        else:
            logits = model(tensor)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(
        prediction, (frame_bgr.shape[1], frame_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def estimate_centerline(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find the center of broad drivable pixel runs from near to far in frame."""
    height, width = mask.shape
    min_run = max(4, int(width * 0.06))
    step = max(1, height // 45)
    rows = range(int(height * 0.96), int(height * 0.46), -step)
    raw: list[tuple[int, int]] = []
    previous_x = width / 2

    for y in rows:
        row = mask[y] == 0
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        runs = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_run]
        if not runs:
            continue
        if not raw:
            # Prefer the drivable run containing the camera's forward axis;
            # otherwise take the broadest nearby region.
            centered = [run for run in runs if run[0] <= width / 2 <= run[1]]
            chosen = max(centered or runs, key=lambda run: run[1] - run[0])
        else:
            chosen = min(runs, key=lambda run: abs((run[0] + run[1]) / 2 - previous_x))
        center_x = int((chosen[0] + chosen[1]) / 2)
        raw.append((center_x, y))
        previous_x = center_x

    if len(raw) < 3:
        return []

    # Average neighboring center estimates to reduce pixel-level mask jitter.
    smooth: list[tuple[int, int]] = []
    for i, (_, y) in enumerate(raw):
        nearby = raw[max(0, i - 2):min(len(raw), i + 3)]
        smooth.append((int(round(sum(p[0] for p in nearby) / len(nearby))), y))
    return smooth


def smooth_centerline(current: list[tuple[int, int]],
                      previous: list[tuple[int, int]],
                      current_weight: float) -> list[tuple[int, int]]:
    """Blend path x-positions with the prior frame at matching image rows."""
    if not current or not previous or current_weight >= 1.0:
        return current
    prior_x_by_y = {y: x for x, y in previous}
    result = []
    for x, y in current:
        prior_x = prior_x_by_y.get(y)
        if prior_x is not None:
            x = int(round((1.0 - current_weight) * prior_x + current_weight * x))
        result.append((x, y))
    return result


def make_overlay(frame_bgr: np.ndarray, mask: np.ndarray,
                 centerline: list[tuple[int, int]]) -> np.ndarray:
    class_map = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(CLASS_COLORS_RGB):
        class_map[mask == class_id] = rgb[::-1]  # BGR for OpenCV
    overlay = cv2.addWeighted(frame_bgr, 0.48, class_map, 0.52, 0)

    if centerline:
        pts = np.array([(x, y) for x, y in reversed(centerline)], dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (20, 40, 42), 9, cv2.LINE_AA)
        path_color = (255, 255, 255)  # White path with a dark outline.
        cv2.polylines(overlay, [pts], False, path_color, 4, cv2.LINE_AA)
        cv2.circle(overlay, tuple(pts[-1]), 7, (255, 255, 255), -1, cv2.LINE_AA)
        near_x, near_y = centerline[0]
        far_x, far_y = centerline[min(len(centerline) - 1, max(0, len(centerline) // 3))]
        offset_pct = 100.0 * (near_x - frame_bgr.shape[1] / 2) / frame_bgr.shape[1]
        heading_deg = np.degrees(np.arctan2(near_x - far_x, far_y - near_y))
        path_status = f"PATH PROPOSAL  offset {offset_pct:+.1f}%  heading {heading_deg:+.1f} deg"
        text_color = (255, 255, 255)
    else:
        path_status = "NO CLEAR DRIVABLE PATH FOUND"
        text_color = (70, 175, 255)

    scale = min(1.0, max(0.55, overlay.shape[1] / 1280.0))
    font = 0.58 * scale
    cv2.rectangle(overlay, (10, 10), (min(overlay.shape[1] - 10, int(620 * scale)), int(45 * scale)), (15, 25, 32), -1)
    cv2.putText(overlay, path_status, (int(20 * scale), int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                font, text_color, max(1, int(2 * scale)), cv2.LINE_AA)

    # Keep a visible key on the image so the class palette is easy to read.
    panel_x, panel_y = 10, int(54 * scale)
    row_h = int(23 * scale)
    panel_w = min(int(218 * scale), overlay.shape[1] - 20)
    panel_h = int(28 * scale) + len(CLASS_NAMES) * row_h + int(7 * scale)
    panel = overlay.copy()
    cv2.rectangle(panel, (panel_x, panel_y),
                  (panel_x + panel_w, min(overlay.shape[0] - 1, panel_y + panel_h)),
                  (15, 25, 32), -1)
    overlay = cv2.addWeighted(panel, 0.78, overlay, 0.22, 0)
    cv2.putText(overlay, "SEGMENTATION CLASSES",
                (panel_x + int(8 * scale), panel_y + int(19 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43 * scale, (255, 255, 255),
                max(1, int(scale)), cv2.LINE_AA)
    for class_id, (name, rgb) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_RGB)):
        y = panel_y + int(28 * scale) + class_id * row_h
        color_bgr = tuple(int(c) for c in rgb[::-1])
        swatch = max(7, int(12 * scale))
        cv2.rectangle(overlay, (panel_x + int(8 * scale), y - swatch + 1),
                      (panel_x + int(8 * scale) + swatch, y + 2), color_bgr, -1)
        cv2.putText(overlay, name, (panel_x + int(28 * scale), y + int(3 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (255, 255, 255),
                    max(1, int(scale)), cv2.LINE_AA)
    return overlay


def fit_preview(frame: np.ndarray, max_width: int = 1280,
                max_height: int = 720) -> np.ndarray:
    """Scale only the on-screen preview to fit; leave inference/output untouched."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame, (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def process_image(input_path: Path, output_path: Path, mask_output: Path | None,
                  model, width: int, height: int, device: torch.device) -> None:
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    mask = predict_mask(frame, model, width, height, device)
    path = estimate_centerline(mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), make_overlay(frame, mask, path)):
        raise OSError(f"Could not write overlay: {output_path}")
    if mask_output:
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_output), mask)
    print(f"Saved overlay: {output_path}")
    if mask_output:
        print(f"Saved class-ID mask: {mask_output}")


def process_video(input_path: Path, output_path: Path, mask_output: Path | None,
                  model, width: int, height: int, device: torch.device,
                  path_smoothing: float, preview: bool = False) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 1:
        fps = 20.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create video output: {output_path}")

    mask_writer = None
    if mask_output:
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        mask_writer = cv2.VideoWriter(
            str(mask_output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (frame_width, frame_height), False,
        )

    frame_index = 0
    previous_centerline: list[tuple[int, int]] = []
    previous_frame_time = time.perf_counter()
    try:
        while True:
            frame_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            mask = predict_mask(frame, model, width, height, device)
            path = smooth_centerline(
                estimate_centerline(mask), previous_centerline, path_smoothing
            )
            previous_centerline = path
            output_frame = make_overlay(frame, mask, path)
            if preview:
                now = time.perf_counter()
                instant_fps = 1.0 / max(now - previous_frame_time, 1e-6)
                previous_frame_time = now
                fps_text = f"VIDEO  {instant_fps:.1f} FPS"
                (text_w, text_h), _ = cv2.getTextSize(
                    fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                )
                x = max(10, output_frame.shape[1] - text_w - 28)
                cv2.rectangle(output_frame, (x - 8, 10),
                              (min(output_frame.shape[1] - 1, x + text_w + 8), 45),
                              (15, 25, 32), -1)
                cv2.putText(output_frame, fps_text, (x, 10 + text_h + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("IDD video segmentation and path proposal",
                           fit_preview(output_frame))
            writer.write(output_frame)
            if mask_writer is not None:
                mask_writer.write(mask)
            frame_index += 1
            if preview:
                delay_ms = max(1, int(1000.0 / fps -
                                      (time.perf_counter() - frame_start) * 1000.0))
                if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), 27):
                    print("Preview stopped by user.")
                    break
            if frame_index % 30 == 0:
                print(f"Processed {frame_index} frames")
    finally:
        capture.release()
        writer.release()
        if mask_writer is not None:
            mask_writer.release()
        if preview:
            cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames. Saved overlay video: {output_path}")
    if mask_output:
        print(f"Saved class-ID mask video: {mask_output}")


def process_camera(camera_index: int, model, width: int, height: int,
                   device: torch.device, path_smoothing: float) -> None:
    """Run live camera inference and show the overlay with measured FPS."""
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise OSError(
            f"Could not open camera {camera_index}. Try another index, such as --camera 1."
        )

    previous_centerline: list[tuple[int, int]] = []
    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    print("Live camera running. Press Q in the video window to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            mask = predict_mask(frame, model, width, height, device)
            path = smooth_centerline(
                estimate_centerline(mask), previous_centerline, path_smoothing
            )
            previous_centerline = path
            output = make_overlay(frame, mask, path)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            fps_text = f"LIVE  {smoothed_fps:.1f} FPS"
            (text_w, text_h), _ = cv2.getTextSize(
                fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            x = max(10, output.shape[1] - text_w - 28)
            cv2.rectangle(output, (x - 8, 10),
                          (min(output.shape[1] - 1, x + text_w + 8), 45),
                          (15, 25, 32), -1)
            cv2.putText(output, fps_text, (x, 10 + text_h + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow("IDD live segmentation and path proposal", fit_preview(output))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Image or video file")
    source.add_argument("--camera", type=int, help="Live camera index, usually 0")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/idd_unet_best.pt"))
    parser.add_argument("--output", type=Path, help="Overlay output path; defaults beside input")
    parser.add_argument("--save-mask", action="store_true", help="Also save a grayscale class-ID mask")
    parser.add_argument("--preview", action="store_true",
                        help="Show video inference as it runs (press Q to stop)")
    parser.add_argument("--path-smoothing", type=float, default=0.25,
                        help="Weight for the current frame's path (0 to 1; default: 0.25)")
    parser.add_argument("--inference-scale", type=float, default=1.0,
                        help="Scale checkpoint input resolution for speed (0.5 to 1.0)")
    args = parser.parse_args()
    if not 0.0 <= args.path_smoothing <= 1.0:
        parser.error("--path-smoothing must be between 0 and 1")
    if not 0.5 <= args.inference_scale <= 1.0:
        parser.error("--inference-scale must be between 0.5 and 1.0")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    is_image = args.input is not None and args.input.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.input is not None:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        if is_image:
            output = args.output or args.input.with_name(args.input.stem + "_path_overlay.png")
            mask_output = output.with_name(output.stem.replace("_path_overlay", "") + "_class_ids.png") if args.save_mask else None
        else:
            output = args.output or args.input.with_name(args.input.stem + "_path_overlay.mp4")
            mask_output = output.with_name(output.stem.replace("_path_overlay", "") + "_class_ids.mp4") if args.save_mask else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model, width, height = load_model(args.checkpoint, device)
    if args.inference_scale < 1.0:
        width = max(32, int(round(width * args.inference_scale / 4)) * 4)
        height = max(32, int(round(height * args.inference_scale / 4)) * 4)
    print(f"Loaded checkpoint on {device}. Input size: {width}x{height}.")
    if device.type == "cuda":
        print("CUDA mixed precision and channels-last inference enabled.")
    print("Classes: " + ", ".join(f"{i}={name}" for i, name in enumerate(CLASS_NAMES)))
    print("Path line is a centerline heuristic from predicted drivable pixels.")
    if args.camera is not None:
        if args.output or args.save_mask or args.preview:
            parser.error("--output, --save-mask, and --preview are only for file input")
        print(f"Temporal path smoothing: current-frame weight {args.path_smoothing:.2f}.")
        process_camera(args.camera, model, width, height, device, args.path_smoothing)
    elif not is_image:
        print(f"Temporal path smoothing: current-frame weight {args.path_smoothing:.2f}.")
        process_video(args.input, output, mask_output, model, width, height, device,
                      args.path_smoothing, args.preview)
    else:
        if args.preview:
            parser.error("--preview applies to video files, not still images")
        process_image(args.input, output, mask_output, model, width, height, device)


if __name__ == "__main__":
    main()
