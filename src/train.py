import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.nn as nn

from models.flood_model import FloodPredictionModel
from data.flood_data_module import FloodDataModule

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

@hydra.main(version_base=None, config_path="../configs", config_name="model_config")
def train(config: DictConfig) -> None:
    """
    Train the flood prediction model
    
    Args:
        config: Hydra configuration
    """
    print(OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision('high')
    
    # Set up wandb logger
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        name=f"lr-{config.training.learning_rate:.0e}_{config.model.num_layers}lys_{config.model.num_heads}heads_{config.model.window_size}ws",
        log_model=True
    )
    
    # Initialize data module
    data_module = FloodDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        normalize=True
    )
    
    # Initialize model
    
    if config.training.checkpoint:
        print('\n')
        print(f'Loading pre-trained model: {config.training.checkpoint}')
        print('\n')
        model=FloodPredictionModel.load_from_checkpoint(config.training.checkpoint, config=config, strict=False)
    else:
        print('initializing model...')
        model = FloodPredictionModel(config)
        model.apply(init_weights)
    
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
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        accelerator="auto",  # Use GPU if available
        devices='auto',
        strategy='ddp',
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
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