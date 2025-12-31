import os
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib import TaskType
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def train_improved_model(
    data_root="./datasets/socket_pins",
    image_size=256,
    batch_size=16,
    num_workers=4,
    max_epochs=50,
):
    """Train improved Patchcore model for socket pins"""
    
    logger.info("Initializing improved Patchcore model...")
    
    # Model configuration
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.25,
        num_neighbors=15,
    )
    
    # Data configuration
    datamodule = Folder(
        name="socket_pins_improved",
        root=data_root,
        normal_dir="train/good",
        abnormal_dir="test/defect",
        task=TaskType.SEGMENTATION,
        image_size=(image_size, image_size),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        val_split_mode="from_test",
        val_split_ratio=0.3,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="results/Patchcore_improved/socket_pins/checkpoints",
        filename="best-{epoch:02d}-{pixel_F1Score:.3f}",
        save_top_k=3,
        monitor="pixel_F1Score",
        mode="max",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="pixel_F1Score",
        patience=10,
        mode="max",
        verbose=True,
    )
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir="results/Patchcore_improved/socket_pins",
        name="logs",
    )
    
    # Engine
    engine = Engine(
        callbacks=[
            TQDMProgressBar(),
            checkpoint_callback,
            early_stop_callback,
        ],
        logger=tb_logger,
        max_epochs=max_epochs,
    )
    
    # Train
    logger.info("Starting training...")
    engine.fit(datamodule=datamodule, model=model)
    
    # Test
    logger.info("Testing model...")
    test_results = engine.test(datamodule=datamodule, model=model)
    
    # Export
    logger.info("Exporting model...")
    engine.export(
        model=model,
        export_type="torch",
        ckpt_path=checkpoint_callback.best_model_path,
    )
    
    logger.info(f"Training complete! Best model saved to: {checkpoint_callback.best_model_path}")
    logger.info(f"Test results: {test_results}")
    
    return test_results
if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    results = train_improved_model()