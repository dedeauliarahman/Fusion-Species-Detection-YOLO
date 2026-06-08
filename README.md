# Fusion-Species-Detection-YOLO
> **RGB-Thermal Image Fusion with YOLOv8 for Robust Wildlife Species Detection in Tropical Ecosystems**
> Baluran National Park, East Java, Indonesia

## Overview
This repository contains the official code and resources for our paper on **First use of visible-thermal fusion network approach for robust species monitoring in the tropics**. To our knowledge, this is the **first application** of a visible-thermal fusion network approach for robust wildlife monitoring in a tropical ecosystem setting.

Drone-based RGB and thermal infrared imagery were fused and used to train a YOLOv8 object detection model, enabling reliable detection of wildlife species under challenging tropical conditions including dense vegetation, variable lighting, and partial occlusion.

**Study site:** Baluran National Park, East Java, Indonesia.

## Target Species
The model was trained and evaluated on the following wildlife species:
| Species |
|---|
| Banteng |
| Javan Deer |
| Long-tailed macaque |
| Water buffalo |
| Wild boar |
| Green peacock |


> 📌 A detailed flowchart of the methodology is available in the paper and in [`docs/Flowdiagram.png`](docs/Flowdiagram.png).



## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Main dependencies:
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`
- `Pillow`

---

## Dataset

The fused RGB-thermal dataset used in this study is publicly available on Zenodo:
> 📦 **[Dataset Link — Zenodo]** *(link will be updated upon publication)*