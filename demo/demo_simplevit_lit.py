import lightning as L
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import SimpleViT

from torchmetrics import AUROC, AveragePrecision, F1Score

logger = logging.getLogger(__name__)


class DemoSimpleVitLit(L.LightningModule):
    def __init__(
        self,
        image_size = 256, 
        patch_size = 32,
        channels = 1,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        lr=0.01,
        weight_decay=0.000001,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.net = SimpleViT(
            image_size = image_size,
            patch_size = patch_size,
            channels = channels,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim
        )

        self.criterion = nn.CrossEntropyLoss()
        self.metrics_names = ["auc", "f1", "auprc"]
        self.val_metrics = nn.ModuleList(
            [
                AUROC(task="multiclass", num_classes=num_classes),
                F1Score(task="multiclass", num_classes=num_classes),
                AveragePrecision(task="multiclass", num_classes=num_classes),
            ]
        )
        self.test_metrics = nn.ModuleList(
            [
                AUROC(task="multiclass", num_classes=num_classes),
                F1Score(task="multiclass", num_classes=num_classes),
                AveragePrecision(task="multiclass", num_classes=num_classes),
            ]
        )
        self.metrics = {
            "val": (self.val_metrics, self.metrics_names),
            "test": (self.test_metrics, self.metrics_names),
        }

    def _prepare_data(self, x, y):
        # Do some shape transformation if needed
        return x, y

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self._prepare_data(x, y)

        logits = self(x)

        loss = self.criterion(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        x, y = batch
        x, y = self._prepare_data(x, y)
        logits = self(x)

        loss = self.criterion(logits, y)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        metrics, metrics_names = self.metrics[stage]

        for idx, metric in enumerate(metrics):
            name = metrics_names[idx]
            metric.update(logits, y)
            self.log(
                f"{stage}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True
            )

    def validation_step(self, batch):
        self.evaluate(batch, "val")

    def test_step(self, batch):
        self.evaluate(batch, "test")

    def predict_step(self, batch):
        x, y = batch
        x, y = self._prepare_data(x, y)

        logits = self(x)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return preds, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        logger.info(f"Using {optimizer.__class__.__name__} optimizer")

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }

