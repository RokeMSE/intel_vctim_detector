import os
from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.models import Padim, Patchcore
from anomalib.engine import Engine
from anomalib import TaskType
task_type = TaskType.SEGMENTATION
import anomalib.models.components.sampling.k_center_greedy as k_center
from tqdm.notebook import tqdm
k_center.tqdm = tqdm
import torch
from torchvision.transforms.v2 import Resize
torch.set_float32_matmul_precision('medium')
# When initializing the Engine, replace the default progress bar
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# %%
def train_socket_inspector():    
    model = Patchcore(
        backbone="wide_resnet50_2",  # Consistent with config
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.15,  # Increased for better coverage
        num_neighbors=15,  # Increased for better scoring
    )
    datamodule = Folder(
        name="socket_pins",
        root="./src/bubble_pins/datasets/socket_pins",
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defect"
    )  
    
    
    
    """ checkpoint_callback = ModelCheckpoint(
        dirpath="results/Patchcore/socket_pins/checkpoints",
        filename="best-{epoch:02d}-{image_F1Score:.2f}",
        save_top_k=3,
        monitor="image_F1Score",
        mode="max",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="image_F1Score",
        patience=5,
        mode="max",
    ) """
    
    
    """ model = Padim(
        backbone="resnet34", 
        n_features=448
    )
    datamodule = Folder(
        name="socket_pins",
        root="./datasets/socket_pins",
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defect",
        num_workers=8,
    )  """
    
    engine = Engine(
    callbacks=[TQDMProgressBar()], 
    max_epochs=50, 
    #accelerator="cpu",  # Forces training on CPU
    #devices=1           # Uses 1 CPU core/device
)
    """ callbacks=[checkpoint_callback, early_stop_callback], """
    
    engine.fit(datamodule=datamodule, model=model)
    engine.test(datamodule=datamodule, model=model)
    
    # Export with pixel-level predictions
    engine.export(
        model=model,
        export_type="torch",
        ckpt_path="./results/Patchcore/socket_pins/latest/weights/lightning/model.ckpt",
    )
    print("Model exported to .pt format successfully.")

train_socket_inspector()

# %%
# from anomalib.deploy import TorchInferencer, OpenVINOInferencer
# import cv2
# import matplotlib.pyplot as plt
# import dotenv
# dotenv.load_dotenv()

# # %%
# def inspect_pin(model_path, pin_image_path):
#     inferencer = TorchInferencer(path=model_path, device='cuda')
#     predictions = inferencer.predict(image=pin_image_path)
    
#     # --- HANDLE HEATMAP SHAPE ---
#     anom_map = predictions.anomaly_map
#     if hasattr(anom_map, "cpu"): 
#         anom_map = anom_map.cpu().numpy()
#     anom_map = anom_map.squeeze() # Removes (1, 256, 256) -> (256, 256)

#     # --- HANDLE SCORE FORMATTING ---
#     # Convert Tensor to float using .item()
#     score = predictions.pred_score.item() 

#     if score > 0.5:
#         print("Anomalous")
#     else:
#         print("Good")

#     # Visualize
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
#     # Original Image
#     ax[0].imshow(cv2.cvtColor(cv2.imread(pin_image_path), cv2.COLOR_BGR2RGB))
#     ax[0].set_title("Input Pin")
#     ax[0].axis('off')
    
#     # Anomaly Heatmap
#     ax[1].imshow(anom_map, cmap='inferno') 
#     ax[1].set_title(f"Score: {score:.2f}") # Now valid because 'score' is a float
#     ax[1].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# img_path = "C:/Users/rokeM/Downloads/Intel/Prj1/intel_vctim_detector/src/bubble_pins/datasets/socket_pins/test/defect/"
# for img in os.listdir(img_path):
#     inspect_pin("C:/Users/rokeM/Downloads/Intel/Prj1/intel_vctim_detector/src/bubble_pins/results/Patchcore/socket_pins/latest/weights/torch/model.pt", f"{img_path}{img}")


