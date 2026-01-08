import cv2
import numpy as np
from scipy.spatial import KDTree

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

def reconstruct_grid(coords, tolerance=0.3):
    """Reconstruct full grid from partial detections"""
    if len(coords) < 10:
        return coords
    
    coords = np.array(coords)
    
    # Estimate grid spacing
    tree = KDTree(coords)
    dist, _ = tree.query(coords, k=2)
    pitch_x = pitch_y = np.median(dist[:, 1])
    
    # Find grid alignment by looking at coordinate distributions
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Cluster x and y coordinates to find grid lines
    x_bins = []
    y_bins = []
    
    for x in sorted(np.unique(x_coords)):
        if not x_bins or x - x_bins[-1] > pitch_x * 0.7:
            x_bins.append(x)
        else:
            x_bins[-1] = (x_bins[-1] + x) / 2
    
    for y in sorted(np.unique(y_coords)):
        if not y_bins or y - y_bins[-1] > pitch_y * 0.7:
            y_bins.append(y)
        else:
            y_bins[-1] = (y_bins[-1] + y) / 2
    
    # Generate full grid
    full_grid = []
    for x in x_bins:
        for y in y_bins:
            full_grid.append((int(x), int(y)))
    
    return full_grid

def get_pin_coordinates(binary_img):
    """Finds pin centers and filters noise using geometric constraints."""
    
    # ---------1. HOUGH TRANSFORM 
    circles = cv2.HoughCircles(
        binary_img, 
        cv2.HOUGH_GRADIENT, 
        dp=1,           
        minDist=12,     # Pins are rarely closer than 12px
        param1=50,      
        param2=9,      # HIGHER = Less sensitive, fewer false positives
        minRadius=4,    
        maxRadius=15    
    )
    
    if circles is None:
        return []

    # Get raw coordinates
    raw_coords = np.round(circles[0, :]).astype("int")[:, :2]
    
    # If we have very few points, we can't reliably filter by grid logic
    if len(raw_coords) < 10:
        return [(x, y) for x, y in raw_coords]

    # -------2. CALCULATE GRID PITCH (Global Property)
    # Use KDTree to find the nearest neighbor for every point
    tree = KDTree(raw_coords)
    dist, _ = tree.query(raw_coords, k=2) # k=2 because k=1 is the point itself
    
    # dist[:, 1] is the distance to the nearest neighbor
    neighbor_distances = dist[:, 1]
    
    # The "Pitch" is the median distance between valid pins
    estimated_pitch = np.median(neighbor_distances)
    
    # ----------3. FILTERING LOGIC
    filtered_coords = []
    
    # Define tolerance (e.g., +/- 30% of pitch)
    min_dist_tol = estimated_pitch * 0.6
    max_dist_tol = estimated_pitch * 1.5
    
    for i, (x, y) in enumerate(raw_coords):
        d = neighbor_distances[i]
        
        # Criterion 1: Nearest Neighbor must be within expected grid distance
        # Real pins always have a neighbor at 'pitch' distance.
        # Noise is usually isolated (d > max) or clumped (d < min).
        if d < min_dist_tol or d > max_dist_tol:
            continue
            
        # Criterion 2: Local Density
        # Check radius = 2 * pitch (should catch ~8 neighbors in a grid)
        search_radius = estimated_pitch * 2.5
        neighbors_idx = tree.query_ball_point([x, y], r=search_radius)
        num_neighbors = len(neighbors_idx) - 1 # Exclude self
        
        # A valid pin in a grid usually has 4-8 neighbors depending on position (edge vs center)
        # We set >= 3 to be safe for corner pins, but exclude isolated noise
        if num_neighbors >= 3:
            filtered_coords.append((x, y))
            
    return filtered_coords if filtered_coords else raw_coords.tolist()

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