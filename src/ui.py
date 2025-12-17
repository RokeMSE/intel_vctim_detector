import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import time

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="VCTIMs  Detection", layout="wide")

# Sidebar for controls
st.sidebar.title("Settings")

# A. Model Selection
model_source = st.sidebar.radio("Model Backend", ["PyTorch (.pt)", "OpenVINO"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# B. Input Source Selection
input_source = st.sidebar.radio("Input Source", ["Live Webcam", "Upload Image/Video"])

# --- MODEL LOADING (Cached) ---
@st.cache_resource
def load_model(source_type):
    # Update these paths to match your actual folder structure
    if source_type == "OpenVINO":
        path = './runs/detect/vctim_detector2/weights/best_openvino_model/' 
        print(f"Loading OpenVINO model from {path}...")
    else:
        path = './runs/detect/vctim_detector2/weights/best.pt'
        print(f"Loading PyTorch model from {path}...")
    
    return YOLO(path)

try:
    model = load_model(model_source)
except Exception as e:
    st.error(f"Error loading model: {e}. Check your file paths!")
    st.stop()

# --- PREDICT & ANNOTATE ---
def process_frame(frame, model, conf_thresh):
    # Run Inference
    results = model(frame, conf=conf_thresh, verbose=False)[0]
    
    missing_count = 0
    normal_count = 0
    
    # Count classes
    for box in results.boxes:
        c = int(box.cls)
        label = model.names[c]
        if "missing" in label.lower():
            missing_count += 1
        else:
            normal_count += 1
            
    # Plot results
    annotated_frame = results.plot()
    return annotated_frame, missing_count, normal_count

# --- UI ---
st.title("VCTIM Inspection System")

# Create two columns for layout
col1, col2 = st.columns([3, 1])

# --- LOGIC BRANCHING ---

if input_source == "Live Webcam":
    # --- WEBCAM MODE ---
    with col2:
        st.subheader("Real-Time Stats")
        status_placeholder = st.empty()
        metric_total = st.empty()
        metric_pass = st.empty()
        metric_fail = st.empty()

    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
        start_btn = st.button("Start Inspection")
        stop_btn = st.button("Stop")
    
    if start_btn:
        cap = cv2.VideoCapture(0) # Change index if using external camera
        if not cap.isOpened():
            st.error("Could not open webcam.")
        
        stop_pressed = False
        while cap.isOpened() and not stop_pressed:
            # Check if stop button is clicked via session state or logic
            # (Streamlit buttons reset on rerun -> rely on the loop breaking externally or via UI interaction)
            # A simple way in Streamlit loops is using a placeholder button to break, but use the sidebar stop or rely on the loop.
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process
            annotated_frame, missing, normal = process_frame(frame, model, conf_threshold)
            
            # Display Video
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", width='stretch')
            
            # Update Stats
            metric_total.metric("Objects Detected", missing + normal)
            metric_pass.metric("Normal (PASS)", normal)
            metric_fail.metric("Missing (FAIL)", missing)
            
            if missing > 0:
                status_placeholder.error(f"FAIL: {missing} DEFECTS")
            else:
                status_placeholder.success("✅ SYSTEM NORMAL")
            
            # Tiny sleep to allow UI updates
            time.sleep(0.01)

elif input_source == "Upload Image/Video":
    uploaded_file = st.sidebar.file_uploader("Choose a file...", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
    
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        with col2:
            st.subheader("File Stats")
            file_status = st.empty()
            file_pass = st.empty()
            file_fail = st.empty()

        with col1:
            st.subheader("Inference Result")
            
            if file_type == 'image':
                # 1. Load Image (PIL loads as RGB)
                image = Image.open(uploaded_file)
                frame_rgb = np.array(image)
                
                # 2. CONVERT RGB -> BGR for YOLO Inference
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # 3. Inference (Pass BGR image)
                annotated_bgr, missing, normal = process_frame(frame_bgr, model, conf_threshold)
                
                # 4. Display (Convert BGR result back to RGB for Streamlit)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Processed Image", use_container_width=True)
                
                # Stats
                file_pass.metric("Normal", normal)
                file_fail.metric("Missing", missing)
                if missing > 0:
                    file_status.error("Defects Found!")
                else:
                    file_status.success("Clean Board")
            
            elif file_type == 'video':
                # Save video to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read() # OpenCV reads video as BGR automatically
                    if not ret:
                        break
                    
                    # Inference (Frame is already BGR, so it's correct)
                    annotated_frame, missing, normal = process_frame(frame, model, conf_threshold)
                    
                    # Display (Convert BGR to RGB for Streamlit)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, caption="Video Inference", use_container_width=True)
                    
                    # Live Stats
                    file_pass.metric("Normal", normal)
                    file_fail.metric("Missing", missing)
                    
                cap.release()