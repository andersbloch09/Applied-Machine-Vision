# Applied Machine Vision

Welcome to the **Applied Machine Vision** repository! This repository provides a full pipeline for training, evaluating, and deploying a YOLO-based model for screw classification using computer vision techniques.

## Project Overview

This repository contains several Python scripts and YOLOv8 models to support:

- **Dataset preparation and splitting**
- **Model training and evaluation**
- **Real-time inference using trained models**

## Files and Their Purpose

### 1. `main.py`
**Purpose**:  
Main script to train the YOLO-based object detection model.

**Key Features**:
- Trains a YOLOv8 model (`yolov8n.pt` or `yolov8s.pt`) on the prepared dataset.
- Saves training results, including metrics and model weights, in the `runs/detect/train22` directory.
- Configurable for different training parameters.

---

### 2. `final_test.py`
**Purpose**:  
Script for running trained models in a test setup for evaluation or demonstration.

**Key Features**:
- Accepts images or live input to visualize screw detection results.
- Loads the trained YOLO model and displays prediction bounding boxes.

---

### 3. `dataCollector.py`
**Purpose**:  
Utility to collect and save labeled data for training.

**Key Features**:
- Captures images and stores them in the dataset folder.
- Can be customized to label and store new training samples.

---

### 4. `data_split.py`
**Purpose**:  
Splits the dataset into training and validation sets using an 80/20 split.

**Key Features**:
- Organizes images and labels into `train/val` directories under `datasets/screws/`.

---

### 5. `move.py`
**Purpose**:  
Utility script to manipulate or organize dataset files.

**Key Features**:
- Can be used to move images or labels between folders.

---

## Training and Evaluation

### YOLO Training:
- Training is performed using the `main.py` script with YOLOv8 (`yolov8n.pt`, `yolov8s.pt`) via the Ultralytics framework.

### Evaluation:
- After training, evaluation metrics such as confusion matrix, precision, and recall can be found under `runs/detect/train22`.

---

## Requirements

### Python Version:
- Python 3.8 or higher

### Required Libraries:
Install the following dependencies with pip:

```bash
pip install ultralytics opencv-python torch torchvision matplotlib
```
