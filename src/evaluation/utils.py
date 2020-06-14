import os

import torch

from src.model.utils import get_backbone
from src.utils import load_yaml

_NUM_CLASSES = 200


def get_model_from_checkpoint(experiment_path, device):

    models_path = os.path.join(experiment_path, 'models')
    checkpoint_path = os.path.join(models_path, os.listdir(models_path)[0])
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    weights = ckpt['state_dict']
    weights = {k.split('model.')[-1]: v for k, v in weights.items()}
    weights = {k.split('backbone.')[-1]: v for k, v in weights.items()}

    hparams = load_yaml(os.path.join(experiment_path, 'hparams.yaml'))
    backbone_name = hparams['backbone']
    model = get_backbone(
            model_name=backbone_name,
            num_classes=_NUM_CLASSES,
            use_pretrained=False
    )
    model.load_state_dict(weights, strict=True)
    model.to(device)
    return model
