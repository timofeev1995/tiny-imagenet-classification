from abc import ABC
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch

from src.data.utils import load_class2id_mapping, load_class_names
from src.evaluation.utils import get_model_from_checkpoint


class BaseEvaluator(ABC):
    def __init__(
            self,
            experiment_path: Union[Path, str],
            device: str = 'cpu',
            dataset_path: Optional[Union[str, Path]] = None
    ):
        self.device = device
        self.model = get_model_from_checkpoint(experiment_path, device)
        self.model.eval()
        if dataset_path:
            self.class_names = load_class_names(Path(dataset_path))
            self.class2id = load_class2id_mapping(Path(dataset_path))
            self.id2class = {id_: class_ for class_, id_ in self.class2id.items()}

    def evaluate(self, image: np.ndarray):
        pass

    def batch_evaluate(self, images: np.ndarray):
        pass


class Evaluator(BaseEvaluator):
    def __init__(
            self,
            experiment_path: Union[Path, str],
            device: str = 'cpu',
            dataset_path: Optional[Union[str, Path]] = None
    ):
        super().__init__(experiment_path, device, dataset_path)

    def evaluate(self, image: np.ndarray):
        imagetensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
        imagetensor = imagetensor.permute(0, 3, 1, 2)
        with torch.no_grad():
            logits = self.model(imagetensor)
            predicted_classes = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
            predicted_classes = predicted_classes.cpu().detach().numpy()
        output = (predicted_classes, )

        if self.class2id:
            class_predictions = [
                (self.class_names[self.id2class[i]], pred)
                for i, pred in enumerate(predicted_classes)
            ]
            class_predictions = sorted(class_predictions, key=lambda x: -x[1])
            output = output + (class_predictions, )

        return output

    def batch_evaluate(self, images: np.ndarray):
        raise NotImplementedError(f'Not implemented for {self.__class__.__name__}.')
