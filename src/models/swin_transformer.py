import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import wandb
from torchmetrics import Accuracy, F1Score, Precision, Recall

class FloodPredictionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Use timm's Swin Transformer implementation with custom input channels
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True,
            in_chans=config.model.in_channels,
            num_classes=1  # Binary classification for flood/no flood
        )
        
        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        
        # Log metrics
        preds = torch.sigmoid(y_hat) > 0.5
        acc = self.train_accuracy(preds, y)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        
        # Log metrics
        preds = torch.sigmoid(y_hat) > 0.5
        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1(preds, y)
        precision = self.val_precision(preds, y)
        recall = self.val_recall(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        
        # Log sample predictions to WandB
        if batch_idx == 0:
            # Placeholder for visualizing predictions
            pass
            
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        
        # Log metrics
        preds = torch.sigmoid(y_hat) > 0.5
        acc = self.test_accuracy(preds, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.training.max_epochs,
                eta_min=1e-6
            ),
            'interval': 'epoch',
            'name': 'lr'
        }
        
        return [optimizer], [scheduler] 