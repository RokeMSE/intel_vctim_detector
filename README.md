# Visual Inspection System

## Overview

This application is a unified computer vision tool designed for industrial quality assurance. It consolidates two distinct inspection systems into a single Streamlit interface:

1. **VCTIM Component Detection:** Identifies missing or present VCTIM components on circuit boards.
2. **Socket Pin Defect Detection:** Detects anomalies (bent or broken pins) in CPU sockets.

The system utilizes a microservice architecture via Docker, combining deep learning models (YOLO and Anomalib) with traditional computer vision techniques (OpenCV) for robust defect detection.

## Features

### 1. VCTIM Inspection Module

* **Methodology:** Object Detection using a custom weighted YOLOv11 model.
* **Classes:** `missing_vctim`, `normal`.
* **Output:** Bounding box visualization and counts of missing vs normal components.

### 2. Socket Pin Inspection Module

* **Methodology:** Two-stage pipeline.
* **Stage 1 (Preprocessing):** Uses Adaptive Thresholding and Hough Circle Transform to locate and crop individual pins from the socket image (allows for fine-grained fintuning).
* **Stage 2 (Inference):** Uses an Anomalib-based unsupervised anomaly detection model to score each pin.


* **Output:** Heatmap visualization of defects and statistical counts of good vs. defective pins.

## Project Structure

```text
.
├── main.py              # Main Streamlit application entry point
├── processor.py         # Computer vision logic for socket pin extraction
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration for the application
├── docker-compose.yml   # Service orchestration
├── weights/             # Directory for model weights
│   ├── best.pt          # Trained YOLO model for VCTIM
│   └── socket_model.pt  # Trained Anomalib model for Socket Pins
└── README.md            # Project documentation

```

## Prerequisites

* Docker Desktop or Docker Engine installed.
* A host machine with AVX support (standard on modern CPUs).
* (Optional) NVIDIA GPU with container toolkit installed for faster inference.

## Installation and Setup

### 1. Clone or Extract Repository

Ensure all source files (`main.py`, `processor.py`, `Dockerfile`, etc.) are in the project root.

### 2. Prepare Model Weights

You must place your trained model files in the `weights/` directory before building the container.

* **YOLO Model:** Rename your trained YOLO weights to `best.pt` and place them in `weights/`.
* **Anomaly Model:** Export your Anomalib model to TorchScript or PT format, rename it to `socket_model.pt`, and place it in `weights/`.

### 3. Build and Run via Docker

Open a terminal in the project directory and run:

```bash
docker-compose up --build

```

This command will:

1. Build the Docker image.
2. Install all required dependencies (Ultralytics, Anomalib, OpenCV, Streamlit).
3. Start the Streamlit server.

## Usage

1. Once the Docker container is running, open your web browser.
2. Navigate to `http://localhost:8501`.
3. **Select Inspection Mode:** Use the sidebar to toggle between "VCTIM Detection" and "Socket Pin Defect".
4. **Upload Image:** Upload a clear image of the PCB or Socket (`.jpg`, `.png`).
5. **Run Inspection:** Click the "Run Inspection" button to process the image.
6. **View Results:**
* The processed image with overlays will appear on the right.
* Metrics (Pass/Fail counts) will be displayed below the image.



## Technical Details

* **Framework:** Streamlit
* **Image Processing:** OpenCV
* **Deep Learning:** PyTorch, Ultralytics YOLOv11, Anomalib
* **Containerization:** Docker (Python 3.10 Slim base)

## Troubleshooting

* **CUDA Errors:** If running without a GPU, ensure the code in `main.py` is set to use CPU for inference.
* **Model Not Found:** Verify that `weights/best.pt` and `weights/socket_model.pt` exist and are correctly mapped in the `docker-compose.yml` or `Dockerfile`.
* **Pin Detection Failures:** If the socket pins are not detected, ensure the input image lighting is consistent with the training data, as the preprocessing step relies on contrast for Adaptive Thresholding.