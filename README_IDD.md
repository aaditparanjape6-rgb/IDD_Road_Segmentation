# IDD multi-class road-scene segmentation

This project adapts the PyTorch U-Net approach from the second Kaggle notebook to
IDD's multi-class labels. It predicts one semantic label per pixel. It is not
instance segmentation: two cars receive the same class label.

## 1. Get the correct data

IDD Lite is the best place to start: it is a small dataset with seven broad
Level-1 classes, and the official download already includes segmentation masks.
The binary masks bundled with road-only Kaggle notebooks are not sufficient for
multi-class training. See the [official IDD Lite instructions](https://idd.insaan.iiit.ac.in/evaluation/ncvgip19/).

The official Lite archive is named `idd-lite.tar.gz`. Extract it first. It uses
`*_image.jpg` camera files and matching `*_label.png` masks (plus optional
`*_inst_label.png` files, which this semantic segmentation script ignores).
The archive contains an `idd20k_lite` directory, which the script detects:

```text
PATH_TO_IDD/
  idd20k_lite/
    leftImg8bit/train/<drive>/*_image.jpg
    leftImg8bit/val/<drive>/*_image.jpg
    gtFine/train/<drive>/*_label.png
    gtFine/val/<drive>/*_label.png
```

IDD Lite masks use seven Level-1 class IDs `0..6`; unlabeled pixels with value
`255` are ignored in the loss and metrics. These are broad categories such as
drivable, living things, vehicles, roadside objects, and sky, not individual
classes such as car versus motorcycle. The script also supports full IDD's
Level-3 labels when those masks are available.

## 2. Install dependencies

Use a Python environment with a PyTorch build that matches your machine (CPU or
CUDA), then install:

```bash
pip install numpy pillow opencv-python segmentation-models-pytorch
```

Install a PyTorch build that matches your computer from the
[official PyTorch install selector](https://pytorch.org/get-started/locally/).

## 3. Train

From this project directory:

```bash
python train_idd_multiclass.py --data "PATH_TO_IDD" --epochs 30 --batch-size 4
```

The best checkpoint by validation mean IoU is written to
`outputs/idd_unet_best.pt`. If you run out of GPU memory, lower the batch size
or image dimensions. IDD Lite's images are small; the default input size is
256x128.

For full IDD after making Level-3 masks with the official
[AutoNUE preparation code](https://github.com/AutoNUE/public-code), run with
`--label-level level3`. This mode defaults to 512x288 inputs.

## What changed from the notebook

- One binary output channel became seven IDD Lite class logits (or 27 Level-3
  logits with full IDD).
- Binary cross-entropy became multiclass cross-entropy with `255` as the ignore label.
- Validation reports mean IoU over classes, in addition to loss.
- A pretrained ResNet-34 encoder initializes the U-Net; the final layer is
  trained for IDD classes.

This is a research/learning baseline. A good validation score alone does not
establish safety or readiness for vehicle control.

## Run the model on your own image or video

`run_perception.py` loads the best checkpoint, predicts the seven classes, and
saves a color segmentation overlay. It also estimates a candidate centerline
from predicted drivable pixels (class ID 0):

```powershell
python .\run_perception.py --input "C:\path\to\road_image.jpg"
python .\run_perception.py --input "C:\path\to\road_video.mp4"
python .\run_perception.py --input "C:\path\to\road_video.mp4" --preview
python .\run_perception.py --camera 0 --checkpoint ".\outputs\idd_unet_augmented.pt"
```

Camera mode opens a live preview; video-file mode with `--preview` shows
inference while the saved clip is processed. Press `Q` in the preview window to
stop. The overlay shows measured processing FPS. This is a perception demo, not
a vehicle controller.

By default, outputs are saved beside the input with `_path_overlay` in the
filename. Add `--save-mask` to also save the raw class-ID image or video. The
path is a simple image-space centerline heuristic; it is not a trained planner,
steering controller, or guarantee of a safe drivable trajectory.
