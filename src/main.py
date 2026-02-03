import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import logging
from anomalib.deploy import TorchInferencer, OpenVINOInferencer
from anomalib import TaskType
task_type = TaskType.SEGMENTATION
import processor
import torch
import dotenv
dotenv.load_dotenv()

# --- CONFIG ---
st.set_page_config(page_title="Unified Industrial Inspector", layout="wide")

# --- SIDEBAR: DEVICE & MODE ---
st.sidebar.header("System Settings")
# Device Selection (CPU vs GPU)
device_choice = st.sidebar.radio("Inference Device", ["CPU", "GPU"])
device = "cuda" if device_choice == "GPU" and torch.cuda.is_available() else "cpu"

if device_choice == "GPU" and not torch.cuda.is_available():
    st.sidebar.warning("GPU selected but CUDA not available. Falling back to CPU.")

# --- LOAD MODELS ---
@st.cache_resource
def load_yolo_model(device):
    # Update paths to relative paths for Docker compatibility
    model = YOLO('./src/models/VCTIM/detect/vctim_detector/weights/best.pt')
    model.to(device)
    return model

@st.cache_resource
def load_anomalib_model(device):
    if device == "cpu":
        # Paths to OpenVINO IR files
        ov_path = './src/models/PIN/openvino/latest/weights/model.xml'
        st.sidebar.info("Using OpenVINO for CPU acceleration")
        return OpenVINOInferencer(path=ov_path, device="CPU")
    else:
        # Default Torch Inference for GPU
        torch_path = './src/models/PIN/torch/latest/weights/torch/model.pt'
        return TorchInferencer(path=torch_path, device=device)

# --- INFERENCE FUNCTIONS ---

def run_vctim_inference(model, img, threshold):
    # Pass 'conf=threshold' to filter predictions by confidence
    results = model(img, conf=threshold)
    res = results[0].plot()
    
    # Count stats
    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy()
    names = model.names
    
    missing_count = 0
    normal_count = 0
    
    for cls in classes:
        label = names[int(cls)]
        if label == 'missing_vctim': 
            missing_count += 1
        elif label == 'normal':
            normal_count += 1
            
    return res, missing_count, normal_count

def run_socket_inference(inferencer, img, threshold, progress_callback=None):
    """Run Pipeline: Preprocess -> Crop Pins -> Anomaly Detect (With Progress Tracking)"""
    # 1. Preprocessing
    binary = processor.get_binary_image(img)
    coords = processor.get_pin_coordinates(binary)
    
    if not coords:
        return img, 0, 0, "No pins detected.", []

    # 2. Extract Pins
    pins_data = processor.extract_pins(img, coords)
    total_pins = len(pins_data)
    
    # 3. Anomaly Inference
    defect_count = 0
    good_count = 0
    scores_list = []
    pin_details = [] 
    
    res_img = img.copy()
    
    for i, pin in enumerate(pins_data):
        # --- UPDATE PROGRESS ---
        if progress_callback:
            # Update progress (0.0 to 1.0)
            progress_callback((i + 1) / total_pins)
            
        # Run inference
        pred = inferencer.predict(image=pin['crop'])
        
        # Handle Float vs Tensor
        if hasattr(pred.pred_score, "item"):
            score = pred.pred_score.item()
        else:
            score = float(pred.pred_score)
            
        scores_list.append(score)
        
        cx, cy = pin['coords']
        half_size = 15 
        
        # Determine Status
        is_defect = score > threshold
        
        if is_defect:
            defect_count += 1
            color = (255, 0, 0) # Red
            status_text = "DEFECT"
        else:
            good_count += 1
            color = (0, 255, 0) # Green
            status_text = "OK"
            
        # Draw on main image
        cv2.rectangle(res_img, 
                      (cx - half_size, cy - half_size), 
                      (cx + half_size, cy + half_size), 
                      color, 2)
        
        # Store for Output
        pin_details.append({
            "id": i,
            "crop": pin['crop'],
            "score": score,
            "status": status_text,
            "is_defect": is_defect
        })
    
    avg_score = np.mean(scores_list) if scores_list else 0.0
    return res_img, defect_count, good_count, f"Processed {len(pins_data)} pins. Avg Score: {avg_score:.3f}", pin_details

# --- UI & BULK LOGIC ---

st.title("Inspection System")

st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("Select Inspection Mode", ["VCTIM Detection", "Socket Pin Defect"])

# Initialize threshold variable
threshold = 0.5 
show_crops = False

# Dynamic Sidebar Controls based on Mode
if mode == "Socket Pin Defect":
    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Sensitivity & View")
    threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.01)
    show_crops = st.sidebar.checkbox("Show Pin Details", value=True)

if mode == "VCTIM Detection":
    st.sidebar.subheader("🎛️ VCTIM Settings")
    threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.25, 0.05)
    use_webcam = st.sidebar.checkbox("Use Webcam (Real-time)")

    model_yolo = load_yolo_model(device)

    if use_webcam:
        st.subheader("Real-time VCTIM Detection")
        # Streamlit component for webcam input
        img_file_buffer = st.camera_input("Take a snapshot or use live view")
        
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            res_img, miss, norm = run_vctim_inference(model_yolo, cv2_img, threshold)
            
            col1, col2 = st.columns(2)
            col1.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="Inference Result")
            col2.metric("Missing", miss, delta_color="inverse")
            col2.metric("Normal", norm)

uploaded_files = st.sidebar.file_uploader(
    "Upload Images (Bulk Supported)", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"### Processing {len(uploaded_files)} images in **{mode}** mode")
    progress_bar = st.progress(0)
    
    total_defects = 0
    total_passed = 0
    
    if mode == "VCTIM Detection":
        model_yolo = load_yolo_model(device)
    else:
        model_ad = load_anomalib_model(device)
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with st.expander(f"Image: {uploaded_file.name}", expanded=True):
            col1, col2 = st.columns(2)
            col1.image(img_rgb, caption="Original", width='stretch')
            
            if mode == "VCTIM Detection":
                try:
                    # UPDATED CALL: Passed 'threshold' here
                    res_img, miss, norm = run_vctim_inference(model_yolo, img, threshold)
                    
                    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    col2.image(res_img_rgb, caption=f"Result (Conf: {threshold})", width='stretch')
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Missing", miss, delta_color="inverse")
                    m2.metric("Normal", norm)
                    
                    total_defects += miss
                    total_passed += norm
                except Exception as e:
                    st.error(f"Error: {e}")

            elif mode == "Socket Pin Defect":
                try:
                    # Create a placeholder for the pin progress
                    st.write("Inspecting Pins...")
                    pin_progress = st.progress(0)
                    
                    # Pass the .progress method as the callback
                    res_img, defects, good, msg, pin_details = run_socket_inference(
                        model_ad, 
                        img_rgb, 
                        threshold,
                        progress_callback=pin_progress.progress
                    )
                    
                    # Clear the progress bar after completion (optional)
                    pin_progress.empty()
                    
                    col2.image(res_img, caption=f"Result (Thresh: {threshold})", width='stretch')
                    m1, m2 = st.columns(2)
                    m1.metric("Defects", defects, delta_color="inverse")
                    m2.metric("Good Pins", good)
                    st.caption(msg)
                    
                    total_defects += defects
                    total_passed += good
                    
                    # --- NEW: DISPLAY PIN CROPS ---
                    if show_crops and pin_details:
                        st.divider()
                        st.markdown("#### 🔍 Individual Pin Inspection")
                        
                        # Filter to show defects first if any exist
                        sorted_pins = sorted(pin_details, key=lambda x: x['score'], reverse=True)
                        
                        # Display in a grid of 8 columns
                        cols = st.columns(8)
                        for idx, p in enumerate(sorted_pins):
                            c = cols[idx % 8]
                            
                            # Color border hack using caption/emoji or PIL
                            # Simple way: Emoji status in caption
                            status_icon = "🔴" if p['is_defect'] else "🟢"
                            
                            c.image(p['crop'], width='stretch')
                            c.caption(f"**{status_icon} {p['score']:.2f}**")
                            
                            # Limit display to avoid crashing browser if 1000 pins
                            if idx > 63: 
                                st.caption("... remaining pins hidden for performance ...")
                                break
                                
                except Exception as e:
                    st.error(f"Error: {e}")

        progress_bar.progress((i + 1) / len(uploaded_files))

    st.divider()
    st.subheader("Batch Summary")
    s1, s2 = st.columns(2)
    s1.metric("Total Defects", total_defects, delta_color="inverse")
    s2.metric("Total Passed", total_passed)

else:
    st.info("Upload one or more images to begin bulk inspection.")