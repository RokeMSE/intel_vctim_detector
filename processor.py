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


def remove_duplicates(coords, min_distance=8):
    """
    Remove duplicate or very close detections using clustering.
    For pins closer than min_distance, keep only one (the centroid).
    """
    if len(coords) < 2:
        return coords
    
    coords_array = np.array(coords)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        return coords
    
    tree = KDTree(coords_array)
    
    # Find all pairs within min_distance
    pairs = tree.query_pairs(r=min_distance)
    
    if not pairs:
        # No duplicates found
        return [(int(x), int(y)) for x, y in coords_array]
    
    # Build clusters of overlapping points
    from collections import defaultdict
    clusters = defaultdict(set)
    point_to_cluster = {}
    cluster_id = 0
    
    for i, j in pairs:
        if i in point_to_cluster and j in point_to_cluster:
            # Merge clusters
            cluster_i = point_to_cluster[i]
            cluster_j = point_to_cluster[j]
            if cluster_i != cluster_j:
                clusters[cluster_i].update(clusters[cluster_j])
                for pt in clusters[cluster_j]:
                    point_to_cluster[pt] = cluster_i
                clusters[cluster_j] = set()
        elif i in point_to_cluster:
            clusters[point_to_cluster[i]].add(j)
            point_to_cluster[j] = point_to_cluster[i]
        elif j in point_to_cluster:
            clusters[point_to_cluster[j]].add(i)
            point_to_cluster[i] = point_to_cluster[j]
        else:
            clusters[cluster_id].update([i, j])
            point_to_cluster[i] = cluster_id
            point_to_cluster[j] = cluster_id
            cluster_id += 1
    
    # Add isolated points
    for i in range(len(coords_array)):
        if i not in point_to_cluster:
            clusters[cluster_id].add(i)
            point_to_cluster[i] = cluster_id
            cluster_id += 1
    
    # Compute centroid for each cluster
    unique_coords = []
    for cluster in clusters.values():
        if cluster:
            cluster_points = coords_array[list(cluster)]
            centroid = np.mean(cluster_points, axis=0)
            unique_coords.append((int(round(centroid[0])), int(round(centroid[1]))))
    
    return unique_coords


def infer_missing_pins(coords, max_gap_multiplier=1.6):
    """
    Infers missing pin locations based on the existing grid pattern.
    Only fills in small gaps, doesn't create entire new rows/columns.
    """
    if len(coords) < 10:
        return coords
    
    coords_array = np.array(coords)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        return coords
    
    tree = KDTree(coords_array)
    
    # Estimate grid pitch
    k = min(5, len(coords_array))
    dist, _ = tree.query(coords_array, k=k)
    
    # Handle case where we only have 1 neighbor (k=2 with self)
    if dist.shape[1] > 1:
        neighbor_dists = dist[:, 1]  # First nearest neighbor (not self)
        pitch = np.median(neighbor_dists)
    else:
        return coords
    
    # Validate pitch
    if pitch <= 0 or np.isnan(pitch):
        return coords
    
    # Define search radius and max gap
    search_radius = pitch * 0.45  # Tighter tolerance for "already exists"
    max_gap = pitch * max_gap_multiplier
    
    # Find missing positions
    inferred_positions = []
    
    # Only check 4 cardinal directions to be more conservative
    directions = [
        (pitch, 0),      # Right
        (-pitch, 0),     # Left
        (0, pitch),      # Down
        (0, -pitch),     # Up
    ]
    
    for x, y in coords_array:
        for dx, dy in directions:
            expected_x = x + dx
            expected_y = y + dy
            
            # Check if this position already has a pin nearby
            neighbors = tree.query_ball_point([expected_x, expected_y], r=search_radius)
            
            # If no neighbor found at expected position
            if len(neighbors) == 0:
                # Check if the gap is reasonable
                # Look for pins on the other side to ensure continuity
                far_x = expected_x + dx
                far_y = expected_y + dy
                far_neighbors = tree.query_ball_point([far_x, far_y], r=search_radius * 1.5)
                
                # Only infer if there's continuity
                if len(far_neighbors) > 0:
                    # Check distance to nearest existing pin
                    try:
                        distances, _ = tree.query([[expected_x, expected_y]], k=1)
                        # Handle both scalar and array returns
                        if hasattr(distances, '__len__') and len(distances) > 0:
                            if hasattr(distances[0], '__len__'):
                                dist_val = distances[0][0]
                            else:
                                dist_val = distances[0]
                        else:
                            dist_val = float(distances)
                        
                        if dist_val <= max_gap:
                            inferred_positions.append((int(round(expected_x)), int(round(expected_y))))
                    except (IndexError, TypeError):
                        # Skip this position if query fails
                        continue
    
    # Combine original and inferred
    all_coords = coords + inferred_positions
    
    # Remove duplicates that might have been created
    all_coords = remove_duplicates(all_coords, min_distance=int(pitch * 0.5))
    
    return all_coords


def get_pin_coordinates(binary_img):
    """
    Finds pin centers, filters noise, and infers missing pins conservatively.
    """
    
    # ---------1. HOUGH TRANSFORM 
    circles = cv2.HoughCircles(
        binary_img, 
        cv2.HOUGH_GRADIENT, 
        dp=1,           
        minDist=12,     # Pins are rarely closer than 12px
        param1=50,      
        param2=12,      # HIGHER = Less sensitive, fewer false positives
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
    tree = KDTree(raw_coords)
    dist, _ = tree.query(raw_coords, k=2)
    
    # Handle the indexing safely
    if dist.shape[1] > 1:
        neighbor_distances = dist[:, 1]
    else:
        return [(x, y) for x, y in raw_coords]
    
    estimated_pitch = np.median(neighbor_distances)
    
    # Validate pitch
    if estimated_pitch <= 0 or np.isnan(estimated_pitch):
        return [(x, y) for x, y in raw_coords]
    
    # ----------3. INITIAL FILTERING LOGIC
    filtered_coords = []
    min_dist_tol = estimated_pitch * 0.6
    max_dist_tol = estimated_pitch * 1.5
    
    for i, (x, y) in enumerate(raw_coords):
        d = neighbor_distances[i]
        
        if d < min_dist_tol or d > max_dist_tol:
            continue
            
        search_radius = estimated_pitch * 2.5
        neighbors_idx = tree.query_ball_point([x, y], r=search_radius)
        num_neighbors = len(neighbors_idx) - 1
        
        if num_neighbors >= 3:
            filtered_coords.append((x, y))
    
    # Use filtered coords or fallback to raw if filtering removed too much
    coords_for_inference = filtered_coords if len(filtered_coords) >= 10 else [(x, y) for x, y in raw_coords]
    
    # Remove any Hough duplicates first
    coords_for_inference = remove_duplicates(coords_for_inference, min_distance=int(estimated_pitch * 0.5))
    
    # ----------4. INFER MISSING PINS (CONSERVATIVELY)
    complete_coords = infer_missing_pins(coords_for_inference, max_gap_multiplier=1.6)
    
    return complete_coords


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