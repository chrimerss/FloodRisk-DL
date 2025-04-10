import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.nn as nn

from models.swin_model import FloodSWINModel
from models.cnn_model import FloodCNNModel
from models.diffusion_model import FloodDiffusionModel
from models.cnn_seg_model import FloodSegmentationModel
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

    wandb.config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    
    # Set up wandb logger
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        name=config.logging.run_name,
        log_model=True
    )
    
    # Initialize data module
    data_module = FloodDataModule(
        data_dir=config.data.data_dir,
        image_size= config.data.image_size,
        num_images= config.data.num_images,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        normalize=True
    )
    
    # Initialize model
    
    if config.training.checkpoint:
        print('\n')
        print(f'Loading pre-trained model: {config.training.checkpoint}')
        print('\n')
        model=FloodSWINModel.load_from_checkpoint(config.training.checkpoint, config=config, strict=False)
    else:
        print('initializing model...')
        if config.model.name=='swin':
            model = FloodSWINModel(config)
        elif config.model.name=='unet':
            model= FloodCNNModel(config)
        elif config.model.name=='diffusion':
            model= FloodDiffusionModel(config)
        elif config.model.name=='segmentation':
            model= FloodSegmentationModel(config)
        else:
            raise ValueError('model name has to be in [swin, unet]]')
        # model.apply(init_weights)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.checkpoints_dir,
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