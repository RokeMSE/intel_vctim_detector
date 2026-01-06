import cv2
import numpy as np

def get_binary_image(img):
    """Converts RGB image to Binary using Adaptive Thresholding."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    adaptive_result = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 55, 7
    )
    return adaptive_result

def get_pin_coordinates(binary_img):
    """Finds pin centers using Hough Circle Transform."""
    circles = cv2.HoughCircles(
        binary_img, 
        cv2.HOUGH_GRADIENT, 
        dp=1,           # Inverse ratio of accumulator resolution
        minDist=15,     # Minimum distance between circle centers
        param1=60,      # Upper threshold for edge detection
        param2=10,      # Accumulator threshold for center detection
        minRadius=5,    # Minimum circle radius
        maxRadius=17    # Maximum circle radius
    )
    
    coords = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        coords = [(x, y) for x, y in circles[:, :2]]
    
    # Filter pins based on distance (simple logic from notebook)
    filtered_coords = []
    if coords:
        coords_array = np.array(coords)
        for x, y in coords:
            # Count neighbors within 200px
            distances = np.sqrt(np.sum((coords_array - [x, y])**2, axis=1))
            neighbors = np.sum(distances < 200) - 1 
            # Keep pins that have neighbors (densely packed)
            if neighbors >= 8: 
                filtered_coords.append((x, y))
                
    return filtered_coords if filtered_coords else coords

def extract_pins(img, coords, crop_size=32):
    """Crops individual pins from the image based on coordinates."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # CLAHE for detail enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    pins = []
    h, w = enhanced.shape
    
    for i, (cx, cy) in enumerate(coords):
        half = crop_size // 2
        # Boundary checks
        if (cy - half >= 0 and cx - half >= 0 and 
            cy + half < h and cx + half < w):
            
            crop = enhanced[cy-half:cy+half, cx-half:cx+half]
            # Resize to 256x256 as usually expected by Anomalib models
            crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            # Convert to RGB for inference (Anomalib expects 3 channels usually)
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2RGB)
            pins.append({
                "id": i,
                "crop": crop_rgb,
                "coords": (cx, cy)
            })
            
    return pins