# IDD Road Scene Segmentation

A course project that trains a semantic segmentation model on the Indian Driving Dataset (IDD) Lite and uses the predicted drivable region to draw a simple candidate path over road scenes.

The model labels each pixel with one of seven broad scene classes. The path is an image-space heuristic derived from pixels predicted as drivable; this project does not control a vehicle or establish that a route is safe.

## What the project can do

- Train a U-Net model for the seven IDD Lite Level-1 classes.
- Predict a class mask and color overlay for images.
- Process video or a webcam stream and display segmentation plus a smoothed path proposal.
- Run a Gradio interface for image, uploaded-video, and live-camera demos.
- Replay saved IDD Lite predictions in a small offline HTML visualization.

### IDD Lite class IDs

| ID | Class |
|---:|---|
| 0 | Drivable |
| 1 | Non-drivable |
| 2 | Living things |
| 3 | Vehicles |
| 4 | Roadside objects |
| 5 | Far objects |
| 6 | Sky |

This is **semantic segmentation**, not instance segmentation: separate cars share the same class label. Unlabeled mask pixels (`255`) are ignored during training.

## Important files

| File or folder | Purpose |
|---|---|
| `app.py` | Gradio showcase UI. Loads a trained checkpoint and handles image inference, WebRTC camera frames, and uploaded-video inference. |
| `train_idd_multiclass.py` | Builds the IDD dataset, trains the U-Net, evaluates validation mIoU and class IoU, and saves the best checkpoint. |
| `run_perception.py` | Command-line inference for an image, video, or camera. Writes color overlays, estimates and smooths a path from predicted drivable pixels, and can optionally save class-ID masks. |
| `predict_idd_lite.py` | Runs batch predictions over an IDD split and saves class-ID masks and color overlays. |
| `make_idd_demo_video.py` | Creates a slideshow video from IDD still images; it does not run inference or simulate a moving car. |
| `driving_simulation.html` | Offline replay visualization using the saved sample images and masks in `assets/idd_lite_demo/`. It does not run the model or control a simulated vehicle. |
| `assets/idd_lite_demo/` | Small image/mask pairs used by the offline HTML replay. |
| `README_IDD.md` | Dataset layout, training details, and additional command-line usage notes. |
| `Course Project Report - IDD Road Scene Segmentation.docx` | Course-project report. |
| `requirements.txt` | Python dependencies; the listed PyTorch build is CPU-only. |
| `.gitignore` | Keeps datasets, caches, model checkpoints, and generated media out of Git. |

## Setup

Use Python 3.12 or newer and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs a CPU-only PyTorch build. For NVIDIA GPU support, first install the PyTorch and torchvision build that matches your CUDA setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the other dependencies separately so the CPU build does not replace it:

```powershell
python -m pip install gradio fastrtc static-ffmpeg segmentation-models-pytorch opencv-python-headless imageio-ffmpeg numpy Pillow
```

## Model checkpoint

Checkpoint files are intentionally **not included** in this GitHub repository. After training, the default checkpoint is `outputs/idd_unet_best.pt`. The app also checks for `idd_unet_targeted.pt` or `idd_unet_augmented.pt` in the project root and in `outputs/`.

To use a checkpoint stored elsewhere, set `MODEL_CHECKPOINT` before launching the app:

```powershell
$env:MODEL_CHECKPOINT = "C:\path\to\idd_unet_best.pt"
python app.py
```

## Train on IDD Lite

Download and extract IDD Lite first. Pass the extracted dataset folder (or its parent) to the training script; do not pass the compressed archive itself.

```powershell
python train_idd_multiclass.py --data "C:\path\to\idd-lite" --epochs 30 --batch-size 4
```

The best validation checkpoint is saved to `outputs/idd_unet_best.pt`. See [`README_IDD.md`](README_IDD.md) for the expected folder layout, supported label levels, and training options.

## Run the demos

### Gradio app

Put a supported checkpoint in one of the locations above, then run:

```powershell
$env:GRADIO_SHARE = "true"
python app.py
```

Open `http://127.0.0.1:7860` for the local interface. With `GRADIO_SHARE` enabled, Gradio prints a temporary public link in the terminal. The Python process and laptop must remain awake while using that link.

### Command-line image or video inference

```powershell
python run_perception.py --input "C:\path\to\road_image.jpg" --checkpoint ".\outputs\idd_unet_best.pt"
python run_perception.py --input "C:\path\to\road_video.mp4" --checkpoint ".\outputs\idd_unet_best.pt" --preview
python run_perception.py --camera 0 --checkpoint ".\outputs\idd_unet_best.pt"
```

The script writes an overlay beside the input by default. Add `--save-mask` to save the raw class-ID mask as well. The path proposal is a smoothed centerline heuristic from the predicted drivable pixels; it is not a learned planner or steering controller.

For the desktop preview windows used by camera mode and video `--preview`, install the GUI-enabled `opencv-python` package instead of `opencv-python-headless` from `requirements.txt`.

### Offline replay

Open `driving_simulation.html` in a browser with its neighboring `assets/idd_lite_demo/` folder intact. This replays saved predictions from seven sample frames; it is a visual replay, not live inference or a physics-based vehicle simulation.

## Limitations and safe use

This is an educational perception prototype trained on a small dataset. Its predictions can be wrong, especially on scenes that differ from the training data. The proposed path is only a visualization derived from image pixels. Do not use this project to control a real vehicle or as a safety system.
