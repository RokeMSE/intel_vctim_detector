import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
""" import logging """
""" from anomalib.deploy import TorchInferencer, OpenVINOInferencer
from anomalib import TaskType """
""" task_type = TaskType.SEGMENTATION """
import processor 
import torch
import dotenv
import report_generator
dotenv.load_dotenv()

# --- HELPER: SCAN DIALOG ---
@st.dialog("Scan Unit ID")
def scan_input_dialog(key_to_update):
    st.write("Please scan the barcode now...")
    
    # Autofocus input
    scanned_val = st.text_input("Scanner Input", key=f"scanner_input_{key_to_update}", help="Click here and scan", value="")
    
    if scanned_val:
        st.session_state[key_to_update] = scanned_val
        st.rerun()

# --- SESSION STATE FOR REPORT ---
if 'image_results' not in st.session_state:
    st.session_state.image_results = []
if 'report_ready' not in st.session_state:
    st.session_state.report_ready = False
if 'last_files_key' not in st.session_state:
    st.session_state.last_files_key = None
if 'cached_totals' not in st.session_state:
    st.session_state.cached_totals = {'defects': 0, 'passed': 0}
if 'last_config' not in st.session_state:
    st.session_state.last_config = None

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
    model = YOLO('./models/VCTIM/detect/vctim_detector/weights/best.pt')
    model.to(device)
    return model

""" @st.cache_resource
def load_anomalib_model(device):
    if device == "cpu":
        ov_path = './models/PIN/openvino/latest/weights/model.xml'
        st.sidebar.info("Using OpenVINO for CPU acceleration")
        return OpenVINOInferencer(path=ov_path, device="CPU")
    else:
        torch_path = './models/PIN/torch/latest/weights/torch/model.pt'
        return TorchInferencer(path=torch_path, device=device) """

# --- INFERENCE FUNCTIONS ---
def run_vctim_inference(model, img, threshold):
    results = model(img, conf=threshold)
    res_img = img.copy()  # Create a copy to draw on
    
    boxes = results[0].boxes
    names = model.names
    
    missing_count = 0
    normal_count = 0
    
    # Iterate through each detected box
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = float(box.conf[0])
        
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Set colors (BGR format for OpenCV)
        if label == 'missing_vctim': 
            color = (0, 0, 255)  # Red for missing
            missing_count += 1
        elif label == 'normal':
            normal_count += 1
            color = (255, 0, 0)  # Blue default
            
        # Draw box and label
        cv2.rectangle(res_img, (x1, y1), (x2, y2), color, 4)
        label_text = f"{label} {conf:.2f}"
        
        # Calculate text size for background
        (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 2)
        cv2.rectangle(res_img, (x1, y1 - h - baseline - 10), (x1 + w + 4, y1), color, -1)
        cv2.putText(res_img, label_text, (x1 + 2, y1 - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)
            
    return res_img, missing_count, normal_count



# def run_socket_inference_batch_optimized(inferencer, img, threshold, progress_callback=None):
#     binary = processor.get_binary_image(img)
#     coords = processor.get_pin_coordinates(binary)
#     
#     if not coords:
#         return img, 0, 0, "No pins detected.", []
# 
#     batch_crops, pins_metadata = processor.extract_pins_batch_optimized(img, coords)
#     
#     if len(batch_crops) == 0:
#         return img, 0, 0, "No valid pins after filtering.", []
#     
#     total_pins = len(batch_crops)
#     try:
#         # GPU / CPU logic
#         if device == "cuda":
#             batch_size = 64
#             # We pass the custom update_interval to the batch function
#             scores_list = run_batch_inference_gpu(inferencer, batch_crops, batch_size, progress_callback)
#         else:
#             batch_size = 32
#             scores_list = run_batch_inference_cpu(inferencer, batch_crops, batch_size, progress_callback)
#             
#     except Exception as e:
#         st.warning(f"Batch inference failed, falling back to sequential: {e}")
#         return run_socket_inference_sequential(inferencer, img, coords, threshold, progress_callback)
#     
#     # 4. Post-processing and visualization
#     defect_count = 0
#     good_count = 0
#     pin_details = []
#     res_img = img.copy()
#     
#     for i, (score, metadata) in enumerate(zip(scores_list, pins_metadata)):
#         cx, cy = metadata['coords']
#         half_size = 15
#         
#         is_defect = score > threshold
#         
#         if is_defect:
#             defect_count += 1
#             color = (255, 0, 0)  # Red
#             status_text = "DEFECT"
#         else:
#             good_count += 1
#             color = (0, 255, 0)  # Green
#             status_text = "OK"
#         
#         # Draw on main image
#         cv2.rectangle(res_img, 
#                      (cx - half_size, cy - half_size), 
#                      (cx + half_size, cy + half_size), 
#                      color, 2)
#         
#         # Store for output (but don't store crops to save memory)
#         pin_details.append({
#             "id": metadata['id'],
#             "crop": batch_crops[i],  # Only if needed for display
#             "score": score,
#             "status": status_text,
#             "is_defect": is_defect
#         })
#     
#     avg_score = np.mean(scores_list) if scores_list else 0.0
#     msg = f"Processed {total_pins} pins. Avg Score: {avg_score:.3f}"
#     
#     return res_img, defect_count, good_count, msg, pin_details
# 
# 
# def run_batch_inference_gpu(inferencer, batch_crops, batch_size, progress_callback=None):
#     scores_list = []
#     total_pins = len(batch_crops)
#     
#     for i in range(0, total_pins, batch_size):
#         batch = batch_crops[i:i+batch_size]
#         
#         # Stack batch into single tensor (REAL batching)
#         batch_np = np.stack(batch)  # Shape: (batch_size, 256, 256, 3)
#         
#         # Call inferencer once per batch, not per image
#         # Note: You may need to modify this depending on Anomalib version
#         try:
#             # Attempt vectorized inference
#             predictions = inferencer.predict(image=batch_np)
#             scores = [p.item() for p in predictions.pred_score]
#         except:
#             # Fallback to sequential if batch fails
#             scores = [inferencer.predict(image=img).pred_score.item() 
#                      for img in batch]
#         
#         scores_list.extend(scores)
#         
#         if progress_callback and i % (batch_size * 5) == 0:  # Update less frequently
#             progress_callback(min((i + batch_size) / total_pins, 1.0))
#     
#     return scores_list
# 
# 
# def run_batch_inference_cpu(inferencer, batch_crops, batch_size, progress_callback=None):
#     """
#     CPU/OpenVINO batch inference
#     """
#     scores_list = []
#     total_pins = len(batch_crops)
#     for i in range(0, total_pins, batch_size):
#         batch = batch_crops[i:i+batch_size]
#         
#         # OpenVINO can handle batches efficiently
#         for crop in batch:
#             pred = inferencer.predict(image=crop)
#             score = pred.pred_score.item() if hasattr(pred.pred_score, "item") else float(pred.pred_score)
#             scores_list.append(score)
#         
#         # Update progress
#         if progress_callback:
#             progress_callback(min((i + batch_size) / total_pins, 1.0))
#     
#     return scores_list
# 
# 
# def run_socket_inference_sequential(inferencer, img, coords, threshold, progress_callback=None):
#     """
#     FALLBACK: Original sequential method (kept for compatibility)
#     """
#     pins_data = processor.extract_pins(img, coords)
#     total_pins = len(pins_data)
#     
#     defect_count = 0
#     good_count = 0
#     scores_list = []
#     pin_details = [] 
#     res_img = img.copy()
#     
#     for i, pin in enumerate(pins_data):
#         if progress_callback:
#             progress_callback((i + 1) / total_pins)
#         
#         pred = inferencer.predict(image=pin['crop'])
#         
#         if hasattr(pred.pred_score, "item"):
#             score = pred.pred_score.item()
#         else:
#             score = float(pred.pred_score)
#         
#         scores_list.append(score)
#         cx, cy = pin['coords']
#         half_size = 15 
#         is_defect = score > threshold
#         
#         if is_defect:
#             defect_count += 1
#             color = (255, 0, 0)
#             status_text = "DEFECT"
#         else:
#             good_count += 1
#             color = (0, 255, 0)
#             status_text = "OK"
#         
#         cv2.rectangle(res_img, 
#                      (cx - half_size, cy - half_size), 
#                      (cx + half_size, cy + half_size), 
#                      color, 2)
#         
#         pin_details.append({
#             "id": i,
#             "crop": pin['crop'],
#             "score": score,
#             "status": status_text,
#             "is_defect": is_defect
#         })
#     
#     avg_score = np.mean(scores_list) if scores_list else 0.0
#     return res_img, defect_count, good_count, f"Processed {len(pins_data)} pins. Avg Score: {avg_score:.3f}", pin_details


# --- UI & BULK LOGIC ---
st.title("Inspection System")

st.sidebar.header("Configuration")
# mode = st.sidebar.selectbox("Select Inspection Mode", ["VCTIM Detection", "Socket Pin Defect"])
mode = "VCTIM Detection" # Socket Pin on hold

threshold = 0.5 
show_crops = False

# Dynamic Sidebar Controls based on Mode
# if mode == "Socket Pin Defect":
#     st.sidebar.divider()
#     st.sidebar.subheader("🎛️ Sensitivity & View")
#     threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.01)
#     show_crops = st.sidebar.checkbox("Show Pin Details", value=True)
#     
#     # Performance options
#     st.sidebar.divider()
#     st.sidebar.subheader("⚡ Performance")
#     use_batch_inference = st.sidebar.checkbox("Use Batch Inference (Faster)", value=True, 
#                                                help="Process pins in batches for 10-50x speedup")

if mode == "VCTIM Detection":
    st.sidebar.subheader("🎛️ VCTIM Settings")
    threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.25, 0.05)
    expected_bib = st.sidebar.number_input("Expected DUT Amount", min_value=1, max_value=20, value=10, step=1, help="Total number of VCTIMs (Normal + Missing) expected per unit")
    use_webcam = st.sidebar.checkbox("Use Webcam (Real-time)")
    model_yolo = load_yolo_model(device)

    if use_webcam:
        st.subheader("Real-time VCTIM Detection")
        img_file_buffer = st.camera_input("Take a snapshot or use live view")
        
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            res_img, miss, norm = run_vctim_inference(model_yolo, cv2_img, threshold)
            
            # Validation
            total_found = miss + norm
            if total_found != expected_bib:
                st.error(f"⚠️ Mismatch! Expected {expected_bib} items, found {total_found} (Miss: {miss}, Norm: {norm})")
            else:
                st.success(f"✅ Count Matches: {total_found}")
            
            col1, col2 = st.columns(2)
            col1.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="Inference Result")
            col2.metric("Missing", miss, delta_color="inverse")
            col2.metric("Normal", norm)

uploaded_files = st.sidebar.file_uploader(
    "Upload Images (JPG/PNG)", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    current_files_key = tuple((f.name, f.size) for f in uploaded_files)
    
    # Smart Caching: Check config + files
    current_config = {
        'mode': mode,
        'device': device,
        'threshold': threshold,
    }
    # Note: expected_bib excluded from config so we can re-validate without re-inference
    
    # Check if we have valid results for all files (crash recovery)
    results_valid = len(st.session_state.get('image_results', [])) == len(uploaded_files)

    if (current_files_key == st.session_state.last_files_key and 
        current_config == st.session_state.last_config and 
        results_valid):
        # Use cached results
        st.write(f"### Showing cached results for {len(uploaded_files)} images in **{mode}** mode")
        st.info("📋 Results loaded from cache. Upload new files or change settings to re-run inspection.")
        
        total_defects = st.session_state.cached_totals['defects']
        total_passed = st.session_state.cached_totals['passed']
        image_results = st.session_state.image_results
        
        # Display cached results
        for i, result in enumerate(image_results):
            with st.expander(f"Image: {result['filename']}", expanded=True):
                col1, col2 = st.columns(2)
                col1.image(result['original_img'], caption="Original", width='stretch')
                col2.image(result['result_img'], caption="Result", width='stretch')
                
                m1, m2 = st.columns(2)
                m1.metric("Defects" if mode == "Socket Pin Defect" else "Missing", result['defects'], delta_color="inverse")
                m2.metric("Passed" if mode == "Socket Pin Defect" else "Normal", result['passed'])
                
                # Dynamic Validation for Cached Results
                if mode == "VCTIM Detection":
                    total_found = result['defects'] + result['passed']
                    if total_found != expected_bib:
                        st.error(f"⚠️ MISSING VCTIM! DUT Count Mismatch! Expected {expected_bib}, Found {total_found}")
                    else:
                        st.success(f"✅ Count Matches: {total_found}")
                
                st.divider()
                st.markdown("**📝 Report Annotations (Optional)**")
                key_prefix = "pin" if mode == "Socket Pin Defect" else "vctim"
                
                # Unit ID with Scanner
                u_col1, u_col2 = st.columns([3, 1])
                unit_key = f"{key_prefix}_unit_{i}"
                
                # Ensure key exists in session state if not already
                if unit_key not in st.session_state:
                    st.session_state[unit_key] = ""
                    
                with u_col1:
                    unit_id = st.text_input("Unit ID", key=unit_key, placeholder="e.g., UNIT-001")
                with u_col2:
                    if st.button("📷 Scan", key=f"scan_btn_cached_{i}"):
                        scan_input_dialog(unit_key)
                
                comments = st.text_area("Comments", key=f"{key_prefix}_comment_{i}", placeholder="Add notes...")
                
                result['unit_id'] = unit_id
                result['comments'] = comments
    else:
        # Run fresh inference
        st.write(f"### Processing {len(uploaded_files)} images in **{mode}** mode")
        
        # Show performance info for Socket Pin mode
        if mode == "Socket Pin Defect" and use_batch_inference:
            st.info("⚡ Using optimized batch inference for faster processing")
        
        progress_bar = st.progress(0)
        
        total_defects = 0
        total_passed = 0
        image_results = []
        
        if mode == "VCTIM Detection":
            model_yolo = load_yolo_model(device)
        """ else:
            model_ad = load_anomalib_model(device) 
        """
    
        for i, uploaded_file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with st.expander(f"Image: {uploaded_file.name}", expanded=True):
                col1, col2 = st.columns(2)
                col1.image(img_rgb, caption="Original", width='stretch')
                
                if mode == "VCTIM Detection":
                    try:
                        res_img, miss, norm = run_vctim_inference(model_yolo, img, threshold)
                        
                        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                        col2.image(res_img_rgb, caption=f"Result (Conf: {threshold})", width='stretch')
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Missing", miss, delta_color="inverse")
                        m2.metric("Normal", norm)

                        # Validation
                        total_found = miss + norm
                        if total_found != expected_bib:
                            st.error(f"⚠️ MISSING VCTIM! DUT Count Mismatch! Expected {expected_bib}, Found {total_found}")
                        else:
                            st.success(f"✅ Count Matches: {total_found}")
                        
                        total_defects += miss
                        total_passed += norm
                        
                        st.divider()
                        st.markdown("**📝 Report Annotations (Optional)**")
                        
                        # Unit ID with Scanner
                        u_col1, u_col2 = st.columns([3, 1])
                        unit_key = f"vctim_unit_{i}"
                        
                        if unit_key not in st.session_state:
                            st.session_state[unit_key] = ""
                            
                        with u_col1:
                            unit_id = st.text_input("Unit ID", key=unit_key, placeholder="e.g., UNIT-001")
                        with u_col2:
                            if st.button("📷 Scan", key=f"scan_btn_vctim_{i}"):
                                scan_input_dialog(unit_key)

                        comments = st.text_area("Comments", key=f"vctim_comment_{i}", placeholder="Add notes...")
                        
                        image_results.append({
                            'filename': uploaded_file.name,
                            'original_img': img_rgb,
                            'result_img': res_img_rgb,
                            'defects': miss,
                            'passed': norm,
                            'unit_id': unit_id,
                            'comments': comments,
                            'pin_details': None
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")

#                 elif mode == "Socket Pin Defect":
#                     try:
#                         st.write("Inspecting Pins...")
#                         pin_progress = st.progress(0)
#                         
#                         # Use optimized batch inference
#                         if use_batch_inference:
#                             res_img, defects, good, msg, pin_details = run_socket_inference_batch_optimized(
#                                 model_ad, 
#                                 img_rgb, 
#                                 threshold,
#                                 progress_callback=pin_progress.progress
#                             )
#                         else:
#                             # Fallback to sequential if user disabled batch
#                             res_img, defects, good, msg, pin_details = run_socket_inference_sequential(
#                                 model_ad, 
#                                 img_rgb,
#                                 processor.get_pin_coordinates(processor.get_binary_image(img_rgb)),
#                                 threshold,
#                                 progress_callback=pin_progress.progress
#                             )
#                         
#                         pin_progress.empty()
#                         
#                         col2.image(res_img, caption=f"Result (Thresh: {threshold})", width='stretch')
#                         m1, m2 = st.columns(2)
#                         m1.metric("Defects", defects, delta_color="inverse")
#                         m2.metric("Good Pins", good)
#                         st.caption(msg)
#                         
#                         total_defects += defects
#                         total_passed += good
#                         
#                         st.markdown("**📝 Report Annotations (Optional)**")
#                         
#                         # Unit ID with Scanner
#                         u_col1, u_col2 = st.columns([3, 1])
#                         unit_key = f"pin_unit_{i}"
#                         if unit_key not in st.session_state:
#                             st.session_state[unit_key] = ""
#                             
#                         with u_col1:
#                             unit_id = st.text_input("Unit ID", key=unit_key, placeholder="e.g., UNIT-001")
#                         with u_col2:
#                             if st.button("📷 Scan", key=f"scan_btn_pin_{i}"):
#                                 scan_input_dialog(unit_key)
# 
#                         comments = st.text_area("Comments", key=f"pin_comment_{i}", placeholder="Add notes...")
#                         
#                         if show_crops and pin_details:
#                             st.divider()
#                             st.markdown("#### 🔍 Individual Pin Inspection")
#                             sorted_pins = sorted(pin_details, key=lambda x: x['score'], reverse=True)
#                             cols = st.columns(8)
#                             for idx, p in enumerate(sorted_pins):
#                                 c = cols[idx % 8]
#                                 status_icon = "🔴" if p['is_defect'] else "🟢"
#                                 c.image(p['crop'], width='stretch')
#                                 c.caption(f"**{status_icon} {p['score']:.2f}**")
#                                 if idx > 63: 
#                                     st.caption("... remaining pins hidden for performance ...")
#                                     break
#                         
#                         image_results.append({
#                             'filename': uploaded_file.name,
#                             'original_img': img_rgb,
#                             'result_img': res_img,
#                             'defects': defects,
#                             'passed': good,
#                             'unit_id': unit_id,
#                             'comments': comments,
#                             'pin_details': pin_details
#                         })
#                                     
#                     except Exception as e:
#                         st.error(f"Error: {e}")
#                         import traceback
#                         st.code(traceback.format_exc())

            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Update cache
        st.session_state.last_files_key = current_files_key
        st.session_state.last_config = current_config
        st.session_state.cached_totals = {'defects': total_defects, 'passed': total_passed}

    # Common code for both cached and fresh results
    st.session_state.image_results = image_results
    st.session_state.total_defects = total_defects
    st.session_state.total_passed = total_passed
    st.session_state.current_mode = mode
    st.session_state.current_device = device
    st.session_state.report_ready = True
    
    # Batch Summary
    st.divider()
    st.subheader("Batch Summary")
    s1, s2 = st.columns(2)
    s1.metric("Total Defects", total_defects, delta_color="inverse")
    s2.metric("Total Passed", total_passed)
    
    # PDF Report Generation
    st.divider()
    st.subheader("📄 Generate Report")
    
    try:
        # Generate JPEG Report
        jpeg_bytes = report_generator.generate_jpeg_report(
            mode=mode,
            device=device,
            image_results=image_results,
            total_defects=total_defects,
            total_passed=total_passed,
            expected_bib=expected_bib if mode == "VCTIM Detection" else None
        )
        
        # Determine filename
        # Priority: First Unit ID found -> Date/Time
        report_filename = f"inspection_report_{mode.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        for res in image_results:
            if res.get('unit_id'):
                # Sanitize filename
                safe_id = "".join(c for c in res['unit_id'] if c.isalnum() or c in (' ', '-', '_')).strip()
                if safe_id:
                    report_filename = f"{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    break
        
        st.download_button(
            label="🖨️ Download Report (.jpeg)",
            data=jpeg_bytes,
            file_name=report_filename,
            mime="image/jpeg",
            type="primary"
        )
        st.caption("Click the button above to download your inspection report.")
    except Exception as e:
        st.error(f"Error generating report: {e}")

else:
    st.info("Upload one or more images to begin bulk inspection.")