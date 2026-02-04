import torch
import numpy as np
from typing import List, Union
from anomalib.deploy import TorchInferencer, OpenVINOInferencer


class BatchInferencerWrapper:
    """
    Wrapper around Anomalib inferencers to enable true batch processing.
    
    The standard Anomalib predict() method processes one image at a time.
    This wrapper adds batch processing capability for massive speedup.
    """
    
    def __init__(self, inferencer: Union[TorchInferencer, OpenVINOInferencer], 
                 device: str = "cuda"):
        self.inferencer = inferencer
        self.device = device
        self.is_openvino = isinstance(inferencer, OpenVINOInferencer)
        
    def predict_batch(self, images: np.ndarray, batch_size: int = 32) -> List[float]:
        """
        Process multiple images in batches.
        
        Args:
            images: numpy array of shape (N, H, W, 3) - batch of RGB images
            batch_size: number of images to process simultaneously
            
        Returns:
            List of anomaly scores (floats)
        """
        
        if self.is_openvino:
            return self._predict_batch_openvino(images, batch_size)
        else:
            return self._predict_batch_torch(images, batch_size)
    
    def _predict_batch_torch(self, images: np.ndarray, batch_size: int) -> List[float]:
        """
        Optimized batch inference for PyTorch models.
        
        KEY OPTIMIZATION: Process multiple images simultaneously on GPU
        """
        scores = []
        n_images = len(images)
        
        # Access the underlying model
        model = self.inferencer.model
        model.eval()
        
        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                batch = images[i:i+batch_size]
                
                # Preprocessing
                # Convert from (B, H, W, C) to (B, C, H, W) and normalize
                batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
                batch_tensor = batch_tensor / 255.0  # Normalize to [0, 1]
                batch_tensor = batch_tensor.to(self.device)
                
                # Forward pass through model
                # The exact method depends on Anomalib version
                try:
                    # Method 1: Direct model forward (Anomalib 1.0+)
                    predictions = model(batch_tensor)
                    
                    # Extract scores from predictions
                    if hasattr(predictions, 'pred_score'):
                        batch_scores = predictions.pred_score
                    elif isinstance(predictions, dict) and 'pred_score' in predictions:
                        batch_scores = predictions['pred_score']
                    else:
                        # Fallback: assume predictions are scores directly
                        batch_scores = predictions
                    
                    # Convert to list of floats
                    if torch.is_tensor(batch_scores):
                        batch_scores = batch_scores.cpu().numpy().tolist()
                    
                    if isinstance(batch_scores, (int, float)):
                        batch_scores = [batch_scores]
                    
                    scores.extend(batch_scores)
                    
                except Exception as e:
                    # Fallback: sequential processing for this batch
                    print(f"Batch processing failed, using sequential: {e}")
                    for img in batch:
                        pred = self.inferencer.predict(image=img)
                        score = pred.pred_score.item() if hasattr(pred.pred_score, "item") else float(pred.pred_score)
                        scores.append(score)
        
        return scores
    
    def _predict_batch_openvino(self, images: np.ndarray, batch_size: int) -> List[float]:
        """
        Optimized batch inference for OpenVINO models.
        
        OpenVINO supports batch inference but needs proper input preparation.
        """
        scores = []
        n_images = len(images)
        
        # OpenVINO batch processing
        # Note: The exact implementation depends on your OpenVINO model configuration
        
        for i in range(0, n_images, batch_size):
            batch = images[i:i+batch_size]
            
            try:
                # Try batch inference
                # Convert batch to format expected by OpenVINO
                batch_processed = self._preprocess_batch_openvino(batch)
                
                # Run inference
                # This is model-specific - adjust based on your model's input/output
                results = self.inferencer.model.infer_new_request(batch_processed)
                
                # Extract scores from results
                # The key name depends on your model's output layer
                output_key = list(results.keys())[0]  # Usually the first output
                batch_scores = results[output_key].flatten().tolist()
                
                scores.extend(batch_scores[:len(batch)])  # In case batch size doesn't match
                
            except Exception as e:
                # Fallback: sequential processing
                print(f"OpenVINO batch processing failed, using sequential: {e}")
                for img in batch:
                    pred = self.inferencer.predict(image=img)
                    score = pred.pred_score.item() if hasattr(pred.pred_score, "item") else float(pred.pred_score)
                    scores.append(score)
        
        return scores
    
    def _preprocess_batch_openvino(self, batch: np.ndarray) -> dict:
        """
        Preprocess batch for OpenVINO inference.
        
        Returns dict with input layer name as key.
        """
        # Convert from (B, H, W, C) to (B, C, H, W)
        batch_tensor = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)
        batch_tensor = batch_tensor / 255.0  # Normalize
        
        # Get input layer name from model
        input_layer = next(iter(self.inferencer.model.inputs))
        
        return {input_layer: batch_tensor}


# ============================================================================
# SIMPLIFIED BATCH INFERENCE (Alternative approach - more compatible)
# ============================================================================

def simple_batch_inference(inferencer, images: np.ndarray, 
                           batch_size: int = 32,
                           progress_callback=None) -> List[float]:
    """
    Simple batch inference that works with any Anomalib inferencer.
    
    This doesn't achieve true parallelization but organizes the processing
    more efficiently and allows for better progress tracking.
    
    Args:
        inferencer: Anomalib TorchInferencer or OpenVINOInferencer
        images: numpy array (N, H, W, 3)
        batch_size: process in chunks for memory efficiency
        progress_callback: function to call with progress (0.0 to 1.0)
    
    Returns:
        List of anomaly scores
    """
    scores = []
    n_images = len(images)
    
    for i in range(0, n_images, batch_size):
        batch = images[i:i+batch_size]
        
        # Process each image in the batch
        for j, img in enumerate(batch):
            pred = inferencer.predict(image=img)
            score = pred.pred_score.item() if hasattr(pred.pred_score, "item") else float(pred.pred_score)
            scores.append(score)
            
            # Update progress
            if progress_callback:
                current_idx = i + j + 1
                progress_callback(current_idx / n_images)
    
    return scores


# ============================================================================
# PARALLEL CPU INFERENCE (For multi-core CPUs)
# ============================================================================

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def parallel_cpu_inference(inferencer, images: np.ndarray, 
                           n_workers: int = None) -> List[float]:
    """
    Parallel inference using multiple CPU cores.
    
    Best for CPU-only inference where you have many cores.
    WARNING: Each worker needs to load the model, so this has memory overhead.
    
    Args:
        inferencer: Anomalib inferencer
        images: numpy array (N, H, W, 3)
        n_workers: number of parallel workers (default: CPU count)
    
    Returns:
        List of anomaly scores
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    def process_single(img):
        pred = inferencer.predict(image=img)
        return pred.pred_score.item() if hasattr(pred.pred_score, "item") else float(pred.pred_score)
    
    # Use ThreadPoolExecutor for I/O-bound operations
    # Use ProcessPoolExecutor for CPU-bound (but has more overhead)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        scores = list(executor.map(process_single, images))
    
    return scores


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    Examples of how to use the batch inference wrappers.
    """
    
    # Example 1: Using BatchInferencerWrapper (best for GPU)
    from anomalib.deploy import TorchInferencer
    
    inferencer = TorchInferencer(path="model.pt", device="cuda")
    batch_wrapper = BatchInferencerWrapper(inferencer, device="cuda")
    
    # Assume we have 400 pin images of shape (400, 256, 256, 3)
    # scores = batch_wrapper.predict_batch(pin_images, batch_size=64)
    
    # Example 2: Using simple batch inference (most compatible)
    # scores = simple_batch_inference(inferencer, pin_images, batch_size=32)
    
    # Example 3: Using parallel CPU inference
    # scores = parallel_cpu_inference(inferencer, pin_images, n_workers=8)
    
    pass


if __name__ == "__main__":
    print("Batch Inference Module for Anomalib")
    print("=" * 50)
    print("This module provides optimized batch inference for Socket Pin detection")
    print("Expected speedup: 10-100x compared to sequential processing")