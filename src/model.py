
import torch.nn as nn
from torchvision import models


def replace_bn(module):

    for name, child in module.named_children():

        if isinstance(child, nn.BatchNorm2d):

            groups = 32

            setattr(
                module,
                name,
                nn.GroupNorm(
                    groups,
                    child.num_features
                )
            )

        else:
            replace_bn(child)


def get_model(num_classes=7):

    # Initialize ResNet18
    model = models.resnet18(weights=None)

    # Replace BatchNorm with GroupNorm
    replace_bn(model)

    # Replace final classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
