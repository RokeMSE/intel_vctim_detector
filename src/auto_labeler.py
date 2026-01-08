import cv2
import os
import numpy as np
import processor  

# --- CONFIG ---
IMAGE_DIR = "C:/Users/rokeM/Downloads/Intel/Prj1/intel_vctim_detector/src/bubble_pins/data/socket/" 
OUTPUT_DIR = "C:/Users/rokeM/Downloads/Intel/Prj1/intel_vctim_detector/src/bubble_pins/data/labels" 
BOX_SIZE = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_to_yolo_format(cx, cy, img_w, img_h, box_w, box_h):
    """
    YOLO format: class_id x_center y_center width height (Normalized 0-1)
    """
    x_c_norm = cx / img_w
    y_c_norm = cy / img_h
    w_norm = box_w / img_w
    h_norm = box_h / img_h
    
    # Class ID 0 for "pin"
    return f"0 {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}"


print(f"Starting auto-labeling for images in {IMAGE_DIR}...")
processed_count = 0

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue
        
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    h, w = img.shape[:2]
    
    # 1. Run Detection Logic
    binary = processor.get_binary_image(img)
    coords = processor.get_pin_coordinates(binary)
    
    # 2. Generate YOLO Labels
    label_lines = []
    for (cx, cy) in coords:
        line = convert_to_yolo_format(cx, cy, w, h, BOX_SIZE, BOX_SIZE)
        label_lines.append(line)
        
    # 3. Save to .txt file
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(OUTPUT_DIR, label_filename)
    
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))
        
    processed_count += 1
    if processed_count % 10 == 0:
        print(f"Processed {processed_count} images...")

print(f"Done! Labels saved to {OUTPUT_DIR}")