import cv2
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN, MeanShift

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
    """
    Hybrid approach: Detect visible pins + reconstruct grid for obstructed ones.
    """
    
    # Step 1: Detect as many visible pins as possible
    visible_coords = detect_visible_pins(binary_img)
    
    if len(visible_coords) < 10:
        # Not enough pins to establish grid - return what we found
        return visible_coords
    
    # Step 2: Estimate grid parameters from visible pins
    grid_params = estimate_grid_params_robust(visible_coords)
    
    if grid_params is None:
        return visible_coords
    
    # Step 3: Generate expected grid positions
    expected_grid = generate_expected_grid(visible_coords, grid_params, binary_img.shape)
    
    # Step 4: CRITICAL - Only keep grid positions that make sense
    validated_grid = validate_grid_against_image(expected_grid, binary_img, visible_coords)
    
    return validated_grid


def detect_visible_pins(binary_img):
    """
    Multi-method detection to find all VISIBLE pins (even deformed ones).
    """
    all_coords = []
    
    # Method 1: Multiple Hough passes with different sensitivities
    for param2_val in [11, 9, 7]:  # Progressively more permissive
        circles = cv2.HoughCircles(
            binary_img, 
            cv2.HOUGH_GRADIENT, 
            dp=1,           
            minDist=10,
            param1=50,      
            param2=param2_val,
            minRadius=3,    
            maxRadius=18    
        )
        
        if circles is not None:
            for x, y in np.round(circles[0, :]).astype("int")[:, :2]:
                # Check if not duplicate
                if not any(abs(x - ex) < 8 and abs(y - ey) < 8 for ex, ey in all_coords):
                    all_coords.append((x, y))
    
    # Method 2: Connected Components for non-circular shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Very permissive filtering
        if 25 < area < 700 and width < 40 and height < 40:
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            
            if not any(abs(cx - ex) < 8 and abs(cy - ey) < 8 for ex, ey in all_coords):
                all_coords.append((cx, cy))
    
    return all_coords


def estimate_grid_params_robust(coords):
    """
    Robustly estimate grid spacing and orientation using RANSAC-like approach.
    """
    if len(coords) < 15:
        return None
    
    coords_array = np.array(coords)
    
    # Find nearest neighbor distances
    tree = KDTree(coords_array)
    distances, _ = tree.query(coords_array, k=6)  # Up to 5 neighbors
    
    # Get all pairwise distances (excluding self)
    all_neighbor_dists = distances[:, 1:].flatten()
    
    # Use histogram to find dominant spacing
    hist, bin_edges = np.histogram(all_neighbor_dists, bins=50, range=(8, 25))
    peak_idx = np.argmax(hist)
    pitch_estimate = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    
    # Refine by taking median of distances near the peak
    valid_distances = all_neighbor_dists[
        (all_neighbor_dists > pitch_estimate * 0.8) & 
        (all_neighbor_dists < pitch_estimate * 1.2)
    ]
    
    if len(valid_distances) < 10:
        return None
    
    pitch = np.median(valid_distances)
    
    return {
        'pitch': pitch,
        'coords': coords_array
    }


def generate_expected_grid(detected_coords, grid_params, img_shape):
    """
    Generate expected grid positions based on detected pins.
    Uses clustering on X and Y coordinates separately.
    """
    coords_array = np.array(detected_coords)
    pitch = grid_params['pitch']
    h, w = img_shape
    
    # Cluster X coordinates to find column positions
    x_coords = coords_array[:, 0].reshape(-1, 1)
    x_clusters = cluster_1d_coordinates(x_coords, bandwidth=pitch * 0.4)
    
    # Cluster Y coordinates to find row positions  
    y_coords = coords_array[:, 1].reshape(-1, 1)
    y_clusters = cluster_1d_coordinates(y_coords, bandwidth=pitch * 0.4)
    
    # Generate grid from cluster centers
    grid_coords = []
    for x_center in x_clusters:
        for y_center in y_clusters:
            # Boundary check
            if 25 < x_center < w - 25 and 25 < y_center < h - 25:
                grid_coords.append((int(x_center), int(y_center)))
    
    return grid_coords


def cluster_1d_coordinates(coords, bandwidth):
    """
    Cluster 1D coordinates (X or Y) to find grid line positions.
    """
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(coords)
    
    cluster_centers = ms.cluster_centers_.flatten()
    return sorted(cluster_centers)


def validate_grid_against_image(grid_coords, binary_img, detected_coords):
    """
    CRITICAL: Validate grid positions to avoid false positives.
    
    Strategy:
    1. Keep all originally detected pins (high confidence)
    2. For inferred positions, check if location makes sense
    3. Remove positions in obvious non-pin areas (large empty regions)
    """
    
    h, w = binary_img.shape
    validated = []
    detected_set = set(detected_coords)
    
    # Create a "pin probability map" from detected positions
    pin_density_map = create_pin_density_map(detected_coords, (h, w))
    
    for x, y in grid_coords:
        # Check 1: Is this an originally detected pin?
        if (x, y) in detected_set or any(abs(x - dx) < 8 and abs(y - dy) < 8 for dx, dy in detected_coords):
            validated.append((x, y))
            continue
        
        # Check 2: Is this position in a high-density region?
        if pin_density_map[y, x] > 0.3:
            # Check 3: Local texture check - not in large uniform area
            if has_local_texture(binary_img, x, y, window=15):
                validated.append((x, y))
    
    return validated


def create_pin_density_map(coords, img_shape):
    """
    Create a heatmap of where pins are likely to exist based on detected pins.
    """
    h, w = img_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    
    # Create Gaussian blobs around each detected pin
    for x, y in coords:
        # Create a Gaussian centered at this pin
        y_grid, x_grid = np.ogrid[-y:h-y, -x:w-x]
        sigma = 30  # Influence radius
        gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        density_map += gaussian
    
    # Normalize
    if density_map.max() > 0:
        density_map /= density_map.max()
    
    return density_map


def has_local_texture(binary_img, x, y, window=15):
    """
    Check if a location has sufficient local texture variation.
    Helps avoid placing pins in large uniform areas (like the socket hole).
    """
    h, w = binary_img.shape
    half = window // 2
    
    y1, y2 = max(0, y - half), min(h, y + half)
    x1, x2 = max(0, x - half), min(w, x + half)
    
    region = binary_img[y1:y2, x1:x2]
    
    if region.size == 0:
        return False
    
    # Calculate local variance
    variance = np.var(region)
    
    # Calculate edge density
    edges = cv2.Canny(region, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Must have some texture (variance > threshold) or edges
    return variance > 500 or edge_density > 0.05


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