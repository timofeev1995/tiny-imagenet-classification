from torch import nn
import torchvision

from src.utils import get_object


def get_backbone(model_name, num_classes, use_pretrained=True):

    model = get_object(f'torchvision.models.{model_name}', pretrained=use_pretrained)
    if model_name.startswith('resnet'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name.startswith('vgg') or model_name.startswith('mnasnet'):
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)

    elif model_name.startswith('squeezenet'):
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes

    elif model_name.startswith('densenet'):
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    else:
        raise AttributeError(f'Model {model_name} is not implemented yet.')

    return model
