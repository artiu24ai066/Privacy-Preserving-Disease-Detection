
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock


def replace_bn(module):

    for name, child in module.named_children():

        if isinstance(child, nn.BatchNorm2d):

            groups = 32

            setattr(
                module,
                name,
                nn.GroupNorm(groups, child.num_features)
            )

        else:
            replace_bn(child)


# Custom BasicBlock forward pass.
# The important change is:
# out = out + identity
# instead of:
# out += identity

def basic_block_forward(self, x):

    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    # Out-of-place addition for Opacus compatibility
    out = out + identity

    out = self.relu(out)

    return out


def get_model(num_classes=7):

    model = models.resnet18(weights=None)

    # Replace BatchNorm with GroupNorm
    replace_bn(model)

    # Replace BasicBlock forward methods
    for module in model.modules():

        if isinstance(module, BasicBlock):
            module.forward = basic_block_forward.__get__(
                module,
                BasicBlock
            )

    # Replace final classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
