import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from models.flood_model import FloodPredictionModel
from data.flood_data_module import FloodDataModule

@hydra.main(version_base=None, config_path="../configs", config_name="model_config")
def train(config: DictConfig) -> None:
    """
    Train the flood prediction model
    
    Args:
        config: Hydra configuration
    """
    print(OmegaConf.to_yaml(config))
    
    # Set up wandb logger
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        name=f"flood_prediction_{config.model.name}",
        log_model=True
    )
    
    # Initialize data module
    data_module = FloodDataModule(
        h5_file=config.data.h5_file,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        normalize=True
    )
    
    # Initialize model
    model = FloodPredictionModel(config)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=config.logging.save_top_k,
        monitor=config.logging.monitor,
        mode=config.logging.mode
    )
    
    early_stopping_callback = EarlyStopping(
        monitor=config.logging.monitor,
        patience=15,
        mode=config.logging.mode
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",  # Use GPU if available
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        log_every_n_steps=config.logging.log_every_n_steps,
        deterministic=True,  # For reproducibility
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    train() 