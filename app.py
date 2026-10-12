"""Gradio showcase app for the IDD Lite road-scene segmentation model."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

if shutil.which("ffprobe") is None:
    try:
        import static_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffprobe was not found. Add the FFmpeg bin folder to PATH, or install "
            "the project fallback with `python -m pip install static-ffmpeg`."
        ) from exc
    static_ffmpeg.add_paths(weak=True)

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
import torch
from fastrtc import VideoStreamHandler, WebRTC

from run_perception import (
    estimate_centerline,
    load_model,
    make_overlay,
    predict_mask,
    smooth_centerline,
)


def find_checkpoint() -> Path:
    configured = os.environ.get("MODEL_CHECKPOINT")
    candidates = [Path(configured)] if configured else []
    candidates += [
        Path("idd_unet_targeted.pt"),
        Path("idd_unet_augmented.pt"),
        Path("outputs/idd_unet_targeted.pt"),
        Path("outputs/idd_unet_augmented.pt"),
        Path("outputs/idd_unet_best.pt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Model checkpoint not found. Add idd_unet_targeted.pt to the project root, "
        "or set MODEL_CHECKPOINT to its path."
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = find_checkpoint()
MODEL, MODEL_WIDTH, MODEL_HEIGHT = load_model(CHECKPOINT_PATH, DEVICE)
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
INFERENCE_LOCK = threading.Lock()


def scaled_size(scale: float) -> tuple[int, int]:
    width = max(32, int(round(MODEL_WIDTH * scale / 4)) * 4)
    height = max(32, int(round(MODEL_HEIGHT * scale / 4)) * 4)
    return width, height


def process_image_upload(image_rgb: np.ndarray | None, scale: float) -> np.ndarray | None:
    if image_rgb is None:
        return None
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    width, height = scaled_size(scale)
    with INFERENCE_LOCK:
        mask = predict_mask(frame, MODEL, width, height, DEVICE)
    overlay = make_overlay(frame, mask, estimate_centerline(mask))
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def process_webcam_frame(
    image_rgb: np.ndarray | None, scale: float
) -> np.ndarray | None:
    """Run one incoming WebRTC camera frame through the road-scene model."""
    return process_image_upload(image_rgb, scale)


def process_video_upload(video_path: str | None, scale: float):
    if not video_path:
        raise gr.Error("Upload a video first.")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise gr.Error("Could not open that video. Try an MP4, MOV, or AVI file.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 1:
        fps = 20.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width < 1 or frame_height < 1:
        capture.release()
        raise gr.Error("Could not read the video dimensions.")

    temp_dir = tempfile.mkdtemp(prefix="idd-seg-")
    raw_path = Path(temp_dir) / "overlay_raw.mp4"
    output_path = Path(tempfile.gettempdir()) / f"idd_overlay_{uuid.uuid4().hex}.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise gr.Error("Could not create the processed video.")

    width, height = scaled_size(scale)
    previous_centerline: list[tuple[int, int]] = []
    frame_count = 0
    chunk_frames = max(2, int(round(fps * 1.5)))
    chunk_index = 0
    chunk_frame_count = 0
    chunk_raw_path = Path(temp_dir) / f"stream_chunk_{chunk_index:05d}.mp4"
    chunk_writer = cv2.VideoWriter(
        str(chunk_raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (frame_width, frame_height),
    )
    if not chunk_writer.isOpened():
        capture.release()
        writer.release()
        raise gr.Error("Could not create the live preview stream.")

    def emit_chunk(path: Path):
        chunk_ts_path = path.with_suffix(".ts")
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-an", "-c:v", "libx264",
                 "-preset", "ultrafast", "-tune", "zerolatency", "-crf", "24",
                 "-pix_fmt", "yuv420p", "-g", str(chunk_frames),
                 "-mpegts_flags", "+resend_headers", "-f", "mpegts",
                 str(chunk_ts_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            raise gr.Error("Could not encode a live preview video chunk.") from exc
        return chunk_ts_path

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            with INFERENCE_LOCK:
                mask = predict_mask(frame, MODEL, width, height, DEVICE)
            path = smooth_centerline(
                estimate_centerline(mask), previous_centerline, 0.25
            )
            previous_centerline = path
            overlay = make_overlay(frame, mask, path)
            writer.write(overlay)
            chunk_writer.write(overlay)
            frame_count += 1
            chunk_frame_count += 1
            if chunk_frame_count >= chunk_frames:
                chunk_writer.release()
                yield str(emit_chunk(chunk_raw_path)), gr.skip()
                chunk_index += 1
                chunk_frame_count = 0
                chunk_raw_path = Path(temp_dir) / f"stream_chunk_{chunk_index:05d}.mp4"
                chunk_writer = cv2.VideoWriter(
                    str(chunk_raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (frame_width, frame_height),
                )
                if not chunk_writer.isOpened():
                    raise gr.Error("Could not continue the live preview stream.")

        if chunk_frame_count:
            chunk_writer.release()
            yield str(emit_chunk(chunk_raw_path)), gr.skip()
    finally:
        capture.release()
        writer.release()
        chunk_writer.release()

    if frame_count == 0:
        raise gr.Error("No readable frames were found in that video.")

    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(raw_path), "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(output_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise gr.Error("Could not convert the output video for browser playback.") from exc
    yield gr.skip(), str(output_path)


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  color-scheme: light;
  --bg: #f3f5f7;
  --surface: #ffffff;
  --ink: #111b27;
  --ink-2: #3c4a5b;
  --muted: #6a7788;
  --line: #d9dfe6;
  --line-soft: #e8ecf0;
  --accent: #b86e00;
  --accent-soft: #fdf3e0;
  --ok: #1f8a4c;
  --radius: 6px;
}
body, .gradio-container {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
.gradio-container { width: 100% !important; max-width: none !important; padding: 0 32px 40px !important; }
footer, .built-with { display: none !important; }
#app-shell { gap: 0 !important; }
html, body { scrollbar-width: none; }
html::-webkit-scrollbar, body::-webkit-scrollbar { display: none; }

/* Header */
#topbar {
  display: flex; align-items: center; justify-content: center; gap: 24px;
  padding: 24px 0 20px;
}
.brand {
  display: flex; flex-direction: column; align-items: center; gap: 10px;
  width: 100%; min-width: 0; box-sizing: border-box; padding: 18px 28px;
  border: 1px solid var(--line); border-radius: 10px;
  background: var(--surface); text-align: center;
}
.brand-mark {
  flex: none; width: 40px; height: 40px; border-radius: 8px;
  background: var(--ink); display: grid; place-items: center;
}
.brand h1 {
  margin: 0; font-size: 21px; line-height: 1.2; font-weight: 600;
  letter-spacing: -0.015em; color: var(--ink);
}
.brand p { margin: 3px 0 0; font-size: 13px; color: var(--muted); }

/* Settings */
.inference-settings {
  margin: 14px 0 0 !important; border: 1px solid var(--line) !important;
  border-radius: var(--radius) !important; background: var(--surface) !important;
  box-shadow: none !important;
}
.inference-settings .label-wrap { color: var(--ink) !important; font-size: 13px !important; font-weight: 600 !important; }

/* Tabs */
.mode-tabs { margin-top: 22px; border: 0 !important; background: transparent !important; }
.mode-tabs .tab-nav, .mode-tabs .tab-wrapper {
  gap: 2px; padding: 0; border-bottom: 1px solid var(--line) !important; background: transparent !important;
}
.mode-tabs .tab-nav button, .mode-tabs .tab-wrapper button {
  margin: 0 0 -1px; padding: 12px 18px !important; border: 0 !important; border-radius: 0 !important;
  background: transparent !important; color: var(--muted) !important;
  font-size: 14px !important; font-weight: 500 !important; transition: color .15s;
}
.mode-tabs .tab-nav button:hover, .mode-tabs .tab-wrapper button:hover { color: var(--ink) !important; }
.mode-tabs .tab-nav button.selected, .mode-tabs .tab-wrapper button.selected {
  color: var(--ink) !important; font-weight: 600 !important;
  box-shadow: inset 0 -2px var(--accent) !important;
}
.mode-tabs .tabitem { padding: 26px 0 0 !important; border: 0 !important; background: transparent !important; }

/* Section text */
.section-heading { margin: 0 0 4px; color: var(--ink); font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }
.section-copy { margin: 0 0 18px; max-width: 70ch; color: var(--muted); font-size: 13.5px; line-height: 1.55; }

/* Media panels */
.media-grid { align-items: stretch !important; gap: 16px !important; }
.media-card {
  padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface);
}
.media-card .block { border-color: var(--line-soft) !important; border-radius: var(--radius) !important; box-shadow: none !important; }
.media-card label, .media-card .wrap > label, .media-card [data-testid="block-label"] {
  color: var(--ink-2) !important; font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: 0 !important; text-transform: none !important;
}

/* Buttons */
button.primary, .primary {
  min-height: 42px; margin-top: 4px; border: 1px solid var(--ink) !important;
  border-radius: var(--radius) !important; color: #fff !important; background: var(--ink) !important;
  font-size: 14px !important; font-weight: 600 !important; box-shadow: none !important;
  transition: background .15s;
}
button.primary:hover, .primary:hover { background: #253449 !important; }

/* Notes */
.hint-card {
  margin: 0 0 16px; padding: 11px 14px; border: 1px solid #f0dcb4;
  border-left: 3px solid var(--accent); border-radius: var(--radius);
  color: #6b4a10; background: var(--accent-soft); font-size: 13px; line-height: 1.5;
}
#safety-note {
  margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 12.5px;
}

/* Accessibility */
button:focus-visible, input:focus-visible, [role="tab"]:focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }

@media (max-width: 760px) {
  .gradio-container { padding: 0 14px 28px !important; }
  #topbar { padding: 18px 0 14px; }
  .mode-tabs .tabitem { padding-top: 18px !important; }
}
"""

BRAND_MARK = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M8 3 3.5 21M16 3l4.5 18" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
    '<path d="M12 4v3M12 10.5v3M12 17v3" stroke="#e9a13b" stroke-width="2" stroke-linecap="round"/>'
    "</svg>"
)

with gr.Blocks(
    title="Road Perception Studio · IDD",
    theme=gr.themes.Base(
        primary_hue="slate", secondary_hue="amber", neutral_hue="slate",
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    ),
    css=APP_CSS,
) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            f"""
            <header id="topbar">
              <div class="brand">
                <div class="brand-mark">{BRAND_MARK}</div>
                <div>
                  <h1>Road scene segmentation</h1>
                  <p>Run the IDD Lite model on an image, a camera stream, or a video.</p>
                </div>
              </div>
            </header>
            """
        )
        with gr.Accordion("Inference settings", open=False, elem_classes="inference-settings"):
            scale = gr.Slider(
                minimum=0.5, maximum=1.0, value=0.75, step=0.05,
                label="Resolution scale (lower is faster, higher preserves detail)",
            )
        with gr.Tabs(elem_classes="mode-tabs"):
            with gr.Tab("Image"):
                gr.HTML(
                    '<h2 class="section-heading">Analyze a road image</h2>'
                    '<p class="section-copy">Upload a scene to view its semantic classes and proposed drivable path.</p>'
                )
                with gr.Row(elem_classes="media-grid"):
                    with gr.Column(elem_classes="media-card"):
                        image_input = gr.Image(
                            type="numpy", label="Input image", height=480,
                            sources=["upload"],
                        )
                    with gr.Column(elem_classes="media-card"):
                        image_output = gr.Image(
                            type="numpy", label="Segmentation and path", height=480,
                        )
                image_button = gr.Button("Analyze image", variant="primary")
                image_button.click(
                    process_image_upload, inputs=[image_input, scale], outputs=image_output
                )
            with gr.Tab("Live camera"):
                gr.HTML(
                    '<h2 class="section-heading">Live road-scene inference</h2>'
                    '<p class="section-copy">Start the camera stream to see segmentation and path output. Camera permission is required.</p>'
                    '<div class="hint-card">For meaningful results, point the camera at a road scene. This model was trained on road imagery.</div>'
                )
                live_camera = WebRTC(
                    label="Live segmentation",
                    mode="send-receive",
                    modality="video",
                    full_screen=False,
                )
                live_camera.stream(
                    fn=VideoStreamHandler(process_webcam_frame, skip_frames=True),
                    inputs=[live_camera, scale],
                    outputs=[live_camera],
                    time_limit=300,
                    concurrency_limit=1,
                )
            with gr.Tab("Video"):
                gr.HTML(
                    '<h2 class="section-heading">Process a road video</h2>'
                    '<p class="section-copy">Watch the inference preview as frames are processed, then review the complete output.</p>'
                )
                with gr.Column(elem_classes="media-card"):
                    video_input = gr.Video(
                        sources=["upload"], label="Input video", height=360,
                    )
                video_button = gr.Button("Process video", variant="primary")
                with gr.Row(elem_classes="media-grid"):
                    with gr.Column(elem_classes="media-card"):
                        video_preview = gr.Video(
                            label="Live inference preview",
                            streaming=True,
                            autoplay=True,
                            height=360,
                        )
                    with gr.Column(elem_classes="media-card"):
                        video_output = gr.Video(
                            label="Processed video", height=360,
                        )
                video_button.click(
                    process_video_upload,
                    inputs=[video_input, scale],
                    outputs=[video_preview, video_output],
                )
        gr.Markdown(
            "Demo for research and education only. The path proposal is not a vehicle controller or a safety system.",
            elem_id="safety-note",
        )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"},
    )
