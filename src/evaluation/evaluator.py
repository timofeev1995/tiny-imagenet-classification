from abc import ABC
from pathlib import Path
from typing import Union

import numpy as np
import torch

from src.evaluation.utils import get_model_from_checkpoint


class BaseEvaluator(ABC):
    def __init__(self, checkpoint: Union[Path, str], device: str = 'cpu'):
        self.device = device
        self.model = get_model_from_checkpoint(checkpoint, device)

    def evaluate(self, image: np.ndarray):
        pass

    def batch_evaluate(self, images: np.ndarray):
        pass


class Evaluator(BaseEvaluator):
    def __init__(self, checkpoint: Union[Path, str], device: str = 'cpu'):
        super().__init__(checkpoint, device)

    def evaluate(self, image: np.ndarray):
        imagetensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
        logits: torch.Tensor = self.model(imagetensor)
        predicted_classes = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
        predicted_classes = predicted_classes.cpu().detach().numpy()
        return predicted_classes

    def batch_evaluate(self, images: np.ndarray):
        raise NotImplementedError(f'Not implemented for {self.__class__.__name__}.')
