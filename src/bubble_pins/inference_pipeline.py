import cv2
import torch
import numpy as np
from ultralytics import YOLO
from anomalib.deploy import TorchInferencer

class SocketInspector:
    def __init__(self, yolo_path, anomalib_path, device='cuda'):
        self.device = device
        # Load YOLO for Localization
        self.yolo = YOLO(yolo_path)
        
        # Load Anomalib for Defect Detection
        # Note: In production, you typically export Anomalib to OpenVINO or ONNX
        self.anomaly_model = TorchInferencer(
            path=anomalib_path,
            device=device
        )

    def process_frame(self, image_path):
        # 1. Read Image
        img = cv2.imread(image_path)
        
        # 2. Localize Socket (YOLO)
        results = self.yolo(img)
        
        # Assuming 1 socket per image, get the box with highest conf
        if len(results[0].boxes) == 0:
            return None, "No Socket Found"
            
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        
        # Crop the socket
        cropped_socket = img[y1:y2, x1:x2]
        
        # 3. Detect Defects (Anomalib)
        # The inferencer handles resizing/normalization internally based on config
        anomaly_map = self.anomaly_model.predict(image=cropped_socket)
        
        # anomaly_map contains 'pred_score', 'pred_label', 'anomaly_map'
        is_defective = anomaly_map.pred_score > 0.5  # Threshold needs tuning
        
        return {
            "original_box": box,
            "cropped_image": cropped_socket,
            "anomaly_score": anomaly_map.pred_score,
            "heatmap": anomaly_map.anomaly_map,
            "is_defective": is_defective
        }

if __name__ == "__main__":
    inspector = SocketInspector(
        yolo_path='weights/yolo_socket.pt',
        anomalib_path='weights/anomalib_model.pt' # Exported torch script
    )
    result = inspector.process_frame("data/test/socket_01.jpg")
    print(f"Defect Probability: {result['anomaly_score']}")