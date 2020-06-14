from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import TinyImageNetDataset, collate_fn
from src.data.utils import compose_augmentations
from src.metrics import accuracy
from src.model.utils import get_backbone
from src.model.classification_model import ClassificationModel


_NUM_CLASSES = 200


class ClassifierModule(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()

        self._hparams = hparams

        self._backbone_name = hparams['backbone']
        self._use_pretrained = hparams['use_pretrained']
        self._backbone = get_backbone(
            model_name=self._backbone_name,
            num_classes=_NUM_CLASSES,
            use_pretrained=self._use_pretrained
        )

        self.batch_size = hparams['batch_size']
        self.learning_rate = hparams['learning_rate']
        self.lr_reduce_factor = hparams['lr_reduce_factor']
        self.patience_steps = hparams['patience_steps']
        self.warmup_steps = hparams['warmup_steps']

        self.dataset_folder = hparams['dataset_folder']
        self.augmentations = compose_augmentations(
            augmentations=hparams['augmentations'],
            p=hparams['augmentations_p']
        )

        self._train_dataloader = None
        self._val_dataloader = None
        self._prepare_data()

        self.model = ClassificationModel(self._backbone)

    def _prepare_data(self):
        train_dataset = TinyImageNetDataset(
            path=self.dataset_folder,
            augmentations=self.augmentations,
            is_valid=False
        )
        valid_dataset = TinyImageNetDataset(
            path=self.dataset_folder,
            is_valid=True
        )
        collate_function = collate_fn
        self._train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_function
        )
        self._val_dataloader = DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_function
        )

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):

        images, classes = batch
        _, loss = self.model(images, classes)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        log = {
            'CCELoss/train': loss,
            'Learning-Rate': lr
        }

        # Set up placeholders for valid metrics.
        if self.trainer.global_step == 0:
            log.update(
                {
                    'CCELoss/valid': np.inf,
                    'Top1accuracy': 0,
                    'Top5accuracy': 0
                }
            )

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        images, classes = batch
        logits, loss = self.model(images, classes)

        return {'val_loss': loss, 'logits': logits, 'true_labels': classes}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'].squeeze(0) for x in outputs], dim=0)
        true_labels = torch.cat([x['true_labels'].squeeze(0) for x in outputs], dim=0)

        top1_accuracy, top5_accuracy = accuracy(output=logits, target=true_labels, topk=(1, 5))
        logs = {
            'CCELoss/valid': loss,
            'Top1accuracy': top1_accuracy,
            'Top5accuracy': top5_accuracy,
        }
        return {'val_loss': loss, 'log': logs}

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def configure_optimizers(self):
        parameters = self.model.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self._hparams['w_decay']
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self._hparams['lr_reduce_factor'],
            patience=self._hparams['patience_steps']
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': self.trainer.accumulate_grad_batches,
            'reduce_on_plateau': True,
            'monitor': 'CCELoss/valid'
        }

        return [optimizer], [scheduler]
