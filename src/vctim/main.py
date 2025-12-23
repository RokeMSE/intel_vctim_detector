import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Dict
import psutil
from io import BytesIO
import pandas as pd

# --- 1. LOGGING ---
# This sets up a log file that rotates every 5MB, keeping 3 backups.
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'inference.log'

""" my_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024, backupCount=3, encoding=None, delay=0)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.INFO) """

app_logger = logging.getLogger('root')
app_logger.setLevel(logging.INFO)
""" app_logger.addHandler(my_handler) """

console = logging.StreamHandler()
console.setLevel(logging.INFO)
app_logger.addHandler(console)

# --- 2. CONFIG ---
st.set_page_config(page_title="VCTIM Inspector", layout="wide", page_icon="")

# Initialize Session State for Metrics
if 'total_count' not in st.session_state: st.session_state['total_count'] = 0
if 'fail_count' not in st.session_state: st.session_state['fail_count'] = 0

st.sidebar.header("Settings")
model_type = st.sidebar.radio("Model Backend", ["OpenVINO (CPU Speed)", "PyTorch (GPU/Std)"])
conf_thresh = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
input_source = st.sidebar.radio("Input Source", ["Live Webcam", "Upload File"])

# --- 3. MODEL & STORE LOADER ---
@st.cache_resource
def load_model(backend):
    try:
        if backend == "OpenVINO (CPU Speed)":
            path = "vctim/runs/detect/vctim_detector3/weights/best_openvino_model/" 
            app_logger.info(f"Loading OpenVINO model from {path}")
            return YOLO(path)
        else:
            path = "vctim/runs/detect/vctim_detector3/weights/best.pt"
            app_logger.info(f"Loading PyTorch model from {path}")
            return YOLO(path)
    except Exception as e:
        app_logger.error(f"Model Load Error: {e}")
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(model_type)

@st.cache_resource
def get_static_store() -> Dict:
    """This dictionary is initialized ONCE and can be used to store the files uploaded"""
    return {}

# --- 4. INFERENCE ENGINE ---
def run_inference(frame):
    start_time = time.time()

    results = model(frame, conf=conf_thresh, verbose=False)[0]

    missing = 0
    normal = 0
    for box in results.boxes:
        c = int(box.cls)
        label = model.names[c]
        if "missing" in label.lower():
            missing += 1
        else:
            normal += 1
            
    # Draw
    annotated = results.plot()
    
    # Log Logic
    latency = (time.time() - start_time) * 1000
    if missing > 0:
        app_logger.warning(f"DEFECT DETECTED: {missing} missing covers. Latency: {latency:.1f}ms")
    
    return annotated, missing, normal, latency

# --- 5. UI LAYOUT ---
tab1, tab2 = st.tabs(["Inspection Dashboard", "System Logs & Monitor"])

with tab1:
    st.title("Production Inspection")
    col_vid, col_stat = st.columns([3, 1])
    
    with col_stat:
        st.subheader("Batch Stats")
        kpi1 = st.metric("Total Inspected", st.session_state['total_count'])
        kpi2 = st.metric("Defects Found", st.session_state['fail_count'], delta_color="inverse")
        status_box = st.empty()

    with col_vid:
        if input_source == "Live Webcam":
            start = st.button("Start Stream")
            stop = st.button("Stop Stream")
            st_frame = st.empty()
            
            if start:
                cap = cv2.VideoCapture(0)
                app_logger.info("Camera started by user.")
                
                while cap.isOpened() and not stop:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_res, miss, norm, lat = run_inference(frame)
                    
                    # Update Stats
                    st.session_state['total_count'] += (miss + norm)
                    st.session_state['fail_count'] += miss
                    
                    # Display
                    st_frame.image(frame_res, channels="BGR", width='stretch')
                    
                    # Status Indicator
                    if miss > 0:
                        status_box.error(f"🚨 FAIL DETECTED ({miss})")
                    else:
                        status_box.success(f"✅ SYSTEM NORMAL ({lat:.0f}ms)")
                    
                cap.release()
                app_logger.info("Camera stopped.")

        elif input_source == "Upload File":
            static_store = get_static_store()
            
            uploaded= st.file_uploader("Upload Image", type=['jpg','png'], accept_multiple_files=True, key="upload")
            if st.button("Clear Uploads"):
                static_store.clear()
                st.session_state.pop('upload', None)  # Clear the uploader widget state too
                st.write("Uploads cleared.")
                st.rerun()

            if uploaded:
                for file in uploaded:
                    static_store[file.name] = file.getvalue()
                st.write(f"Uploaded {len(uploaded)} images.")

            for file in static_store:
                img = Image.open(BytesIO(static_store[file]))
                img = np.array(img)
                # Convert RGB to BGR for YOLO (ESSENTIAL)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                res, miss, norm, lat = run_inference(img)
                
                # Convert back for Display
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                st.image(res, caption="Inference Result")
                app_logger.info(f"Processed {file} Result: {miss} Fail, {norm} Pass")
            

with tab2:
    st.header("System Health Monitor")
    
    # 1. Resource Usage
    c1, c2 = st.columns(2)
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    c1.metric("CPU Usage", f"{cpu}%")
    c2.metric("RAM Usage", f"{ram}%")
    
    """ st.subheader("Application Logs (tail -n 20)")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in reversed(lines[-20:]):
                if "WARNING" in line:
                    st.error(line.strip())
                elif "INFO" in line:
                    st.info(line.strip())
                else:
                    st.text(line.strip())
    else:
        st.warning("No logs found yet.") """

