# Visual Inspection System

## Overview
This application is a unified computer vision tool designed for industrial quality assurance. It consolidates two distinct inspection systems into a single Streamlit interface:
1. **VCTIM Component Detection:** Identifies missing or present VCTIM components on circuit boards using YOLOv11.
2. **Socket Pin Defect Detection:** Detects anomalies (bent or broken pins) in CPU sockets using Anomalib.
The system combines deep learning models (YOLO and Anomalib) with traditional computer vision techniques (OpenCV) for robust defect detection, with support for bulk image processing.

## Features

### 1. VCTIM Inspection Module
* **Methodology:** Object Detection using a custom-trained YOLOv11m model.
* **Classes:** `missing_vctim`, `normal`.
* **Adjustable Confidence Threshold:** Control detection sensitivity with a slider (0.0 - 1.0).
* **Output:** Bounding box visualization and counts of missing vs normal components.
* **Performance:** Achieved 97.6% precision and 96.0% recall on validation data.

### 2. Socket Pin Inspection Module
* **Methodology:** Two-stage pipeline with preprocessing and anomaly detection.
* **Stage 1 (Preprocessing):** 
  - Adaptive Thresholding for contrast enhancement
  - Hough Circle Transform for pin localization
  - Smart duplicate removal and missing pin inference
  - CLAHE enhancement for detail preservation
* **Stage 2 (Inference):** 
  - Anomalib Patchcore-based unsupervised anomaly detection
  - Per-pin scoring with adjustable threshold
  - Real-time progress tracking during inspection
* **Output:** 
  - Visual overlay with color-coded bounding boxes (green = OK, red = defect)
  - Statistical counts of good vs. defective pins
  - Optional detailed pin inspection view showing individual crops with scores

### 3. Bulk Processing
* **Multi-Image Upload:** Process multiple images in a single batch.
* **Progress Tracking:** Visual progress bars for both batch and per-image processing.
* **Batch Summary:** Aggregated statistics across all processed images.
* **Expandable Results:** Each image result in a collapsible section for easy review.

## Project Structure
```text
.
├── src/
│   ├── main.py                    # Main Streamlit application
│   ├── processor.py               # Computer vision pipeline for socket pins
│   ├── vctim/
│   │   ├── data.yml              # YOLO dataset configuration
│   │   ├── Dockerfile            # Docker configuration for VCTIM module
│   │   ├── docker-compose.yml    # Service orchestration
│   │   ├── test_yolo.ipynb       # Training and testing notebook
│   │   └── runs/detect/vctim_detector8/weights/
│   │       └── best.pt           # Trained YOLO model
│   └── bubble_pins/
│       ├── data.yml              # Dataset configuration for socket pins
│       └── results/Patchcore/socket_pins/latest/weights/torch/
│           └── model.pt          # Trained Anomalib model
└── README.md                      # Project documentation
```

## Prerequisites
* Python 3.10 or higher
* CUDA-capable GPU (optional, for faster inference)
* Required Python packages (see Installation)

## Installation and Setup
### 1. Clone or Extract Repository
Ensure all source files are in the project root.

### 2. Install Dependencies
```bash
pip install streamlit ultralytics opencv-python-headless anomalib torch pillow python-dotenv scipy numpy
```

### 3. Prepare Model Weights
You must have trained model files in the following locations:
* **YOLO Model:** `src/vctim/runs/detect/vctim_detector/weights/best.pt`
* **Anomaly Model:** `src/bubble_pins/results/Patchcore/socket_pins/latest/weights/torch/model.pt`

### 4. Environment Configuration (Optional)
Create a `.env` file in the `src/` directory if needed for additional configuration.

### 5. Run the Application
```bash
cd src
streamlit run main.py
```

## Usage
1. Once the application starts, open your web browser to the displayed URL (typically `http://localhost:8501`).

2. **Select Inspection Mode:** Use the sidebar dropdown to choose between:
   - "VCTIM Detection" - for circuit board component inspection
   - "Socket Pin Defect" - for CPU socket pin inspection

3. **Adjust Settings (Sidebar):**
   - **VCTIM Mode:** Set detection confidence threshold (default: 0.25)
   - **Socket Pin Mode:** 
     - Set anomaly threshold (default: 0.5)
     - Toggle "Show Pin Details" to view individual pin crops

4. **Upload Images:** Use the file uploader to select one or more images (`.jpg`, `.png`, `.jpeg`).

5. **View Results:**
   - **Original vs. Processed:** Side-by-side comparison
   - **Metrics:** Pass/Fail counts displayed below images
   - **Pin Details (Socket Mode):** Grid view of individual pins with scores
   - **Batch Summary:** Aggregated statistics at the bottom

6. **Interpret Results:**
   - **VCTIM Detection:** Green boxes = normal components, Red boxes = missing components
   - **Socket Pins:** Green boxes = good pins, Red boxes = defective pins
   - Scores closer to 1.0 indicate higher anomaly likelihood

## Technical Details
### Frameworks & Libraries
* **UI Framework:** Streamlit
* **Image Processing:** OpenCV, NumPy, SciPy
* **Deep Learning:** PyTorch, Ultralytics YOLOv11, Anomalib (Patchcore)
* **Model Deployment:** TorchInferencer (Anomalib)

### Model Specifications
#### VCTIM Detector
* **Architecture:** YOLOv11m (medium)
* **Training:** 100 epochs on custom dataset
* **Input Size:** 640x640
* **Classes:** 2 (missing_vctim, normal)
* **Performance:** 97.1% mAP50, 89.6% mAP50-95

#### Socket Pin Detector
* **Architecture:** Patchcore (Anomalib)
* **Task Type:** Segmentation-based anomaly detection
* **Preprocessing:** 
  - Adaptive thresholding (block size: 55, C: 7)
  - Hough Circle detection (minDist: 12, minRadius: 4, maxRadius: 15)
  - Intelligent duplicate removal and grid inference
* **Input Size:** 256x256 (per pin crop)

### Key Algorithms
#### Pin Detection Pipeline (`processor.py`)
1. **Binary Image Generation:** Gaussian blur + adaptive thresholding
2. **Pin Localization:** Hough Circle Transform with conservative parameters
3. **Duplicate Removal:** KDTree-based clustering with configurable tolerance
4. **Missing Pin Inference:** Grid-based interpolation for incomplete detections
5. **Pin Extraction:** CLAHE enhancement + boundary-safe cropping

## Training the Models
### VCTIM Detector Training
See `src/vctim/test_yolo.ipynb` for the complete training pipeline:

```python
from ultralytics import YOLO

model = YOLO('yolo11m.pt')  # Load pretrained weights
results = model.train(
    data='data.yml',
    epochs=100,
    imgsz=640,
    batch=32,
    name='vctim_detector8'
)
```

**Dataset Structure (data.yml):**
```yaml
train: ./data/train/images
val: ./data/valid/images
test: ./data/test/images
nc: 2
names: ['missing_vctim', 'normal']
```

### Socket Pin Detector Training
Use Anomalib's training pipeline with the Patchcore algorithm on normal pin images for unsupervised learning.

## Docker Deployment (VCTIM Module)
A Dockerfile is provided for the VCTIM inspection module:
```bash
cd src/vctim
docker-compose up --build
```
**Note:** The current main application (`main.py`) runs both modules together and requires both models. For production deployment, you may want to containerize the complete application.

## Troubleshooting
### Common Issues
**Model Not Found Errors:**
- Verify model weights exist at the specified paths
- Check that file paths in `load_yolo_model()` and `load_anomalib_model()` match your directory structure
- Update hardcoded paths to use relative paths if needed

**CUDA/GPU Errors:**
- The application automatically falls back to CPU if CUDA is unavailable
- For CPU-only systems, expect slower inference times
- Check `torch.cuda.is_available()` to verify GPU detection

**Pin Detection Failures:**
- Ensure input images have consistent lighting (adaptive thresholding relies on contrast)
- Adjust Hough Circle parameters in `processor.py` if pins are not detected:
  - `minDist`: Minimum distance between pin centers (default: 12)
  - `param2`: Accumulator threshold for circle detection (default: 12, higher = fewer false positives)
  - `minRadius`/`maxRadius`: Expected pin size range

**Slow Processing:**
- Enable GPU support for faster inference
- Reduce batch size when processing many images
- Disable "Show Pin Details" for faster socket inspection

**Memory Issues:**
- Process images in smaller batches
- Reduce image resolution before upload
- Limit the number of pin details displayed (currently capped at 64)

## Performance Optimization Tips
1. **GPU Acceleration:** Ensure CUDA is properly installed for 5-10x speedup
2. **Batch Size:** Adjust based on available RAM/VRAM
3. **Model Optimization:** Consider exporting models to ONNX or OpenVINO format
4. **Image Preprocessing:** Resize large images before upload to reduce processing time

## Future Enhancements
- [ ] Add export functionality for inspection reports (CSV/PDF)
- [ ] Implement database logging for quality tracking
- [ ] Add support for real-time camera/video stream inspection
- [ ] Create REST API endpoints for integration with manufacturing systems
- [ ] Implement automated retraining pipeline with feedback loop
- [ ] Enhance visualization with heatmaps and statistical charts