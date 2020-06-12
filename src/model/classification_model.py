from typing import Optional

import torch
from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = nn.functional.cross_entropy

    def forward(self, images: torch.Tensor, classes: Optional[torch.Tensor] = None):
        logits = self.backbone(images)
        output = (logits, )
        if classes is not None:
            loss = self.loss_fn(logits, classes)
            output += (loss, )
        return output
