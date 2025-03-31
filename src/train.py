import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from models.swin_transformer import FloodPredictionModel
from data.data_module import FloodDataModule
from utils.visualization import log_to_wandb

@hydra.main(version_base=None, config_path="../configs", config_name="model_config")
def train(config: DictConfig) -> None:
    """
    Train the flood prediction model
    
    Args:
        config: Hydra configuration
    """
    # Set up wandb logger
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        name=f"swin_transformer_flood_prediction",
        log_model=True
    )
    
    # Initialize data module
    data_module = FloodDataModule(config)
    
    # Initialize model
    model = FloodPredictionModel(config)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=config.logging.save_top_k,
        monitor=config.logging.monitor,
        mode=config.logging.mode
    )
    
    early_stopping_callback = EarlyStopping(
        monitor=config.logging.monitor,
        patience=10,
        mode=config.logging.mode
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",  # Use GPU if available
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        log_every_n_steps=config.logging.log_every_n_steps,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Log sample predictions
    log_to_wandb(trainer, model, data_module.val_dataloader(), num_samples=4)
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    train()