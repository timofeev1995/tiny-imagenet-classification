import torch

from src.model.utils import get_backbone


_NUM_CLASSES = 200


def get_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    weights = ckpt['state_dict']
    weights = {k.split('model.')[-1]: v for k, v in weights.items()}

    hparams = ckpt['hparams']
    backbone_name =  hparams['backbone']
    model = get_backbone(
            model_name=backbone_name,
            num_classes=_NUM_CLASSES,
            use_pretrained=False
        )
    model.load_state_dict(weights)
    model.to(device)
    return model
