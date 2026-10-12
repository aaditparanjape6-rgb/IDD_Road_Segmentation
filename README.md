---
title: IDD Road Scene Segmentation
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12"
---

# IDD Road Scene Segmentation Demo

Upload a road image or short road video, or allow webcam access for live frame
inference, to see seven-class IDD Lite semantic segmentation and a simple path
proposal based on predicted drivable pixels.

## Model checkpoint

Add the trained `idd_unet_targeted.pt` file to the repository root before
launching. If it is not available, the app also checks for
`idd_unet_augmented.pt` and `outputs/idd_unet_best.pt`. You can set the
`MODEL_CHECKPOINT` environment variable to use a different checkpoint path.

Do not upload the IDD training dataset to the demo Space; the app only needs the
trained checkpoint. Keep test clips short because video processing can take a
while on CPU hardware. The resolution slider trades fine detail for speed.

This is an educational perception demo, not a self-driving controller or a
safety system.
