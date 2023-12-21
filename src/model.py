import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torchvision.models.segmentation import fcn_resnet50, fcn


class CatDogClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(CatDogClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 8 * 8, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RaceClassifier(nn.Module):
    def __init__(self, num_classes: int = 37):
        super(RaceClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 8 * 8, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class HeadDetection(nn.Module):
    pass


class AnimalSegmentation(nn.Module):
    def __init__(self, num_classes: int = 21):
        super(AnimalSegmentation, self).__init__()
        self.backbone = fcn_resnet50(pretrained=True).backbone
        self.classifier = fcn.FCNHead(2048, num_classes)
        self.aux_classifier = fcn.FCNHead(1024, num_classes)

    def forward(self, x) -> Tensor:
        features = self.backbone(x)
        main_classifier = self.classifier(features["out"])
        aux_classifier = self.aux_classifier(features["aux"])
        return main_classifier, aux_classifier


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.segmentation as segmentation


class AnimalSegmentation2(nn.Module):
    def __init__(self, num_classes: int = 21):
        super(AnimalSegmentation2, self).__init__()
        resnet50 = models.resnet50(pretrained=True)

        # Tomar solo las capas necesarias de ResNet como el "backbone"
        self.backbone = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
        )

        # Añadir capas de "head" y "aux_head" para la clasificación
        self.head = segmentation.fcn.FCNHead(2048, num_classes)
        self.aux_head = segmentation.fcn.FCNHead(1024, num_classes)

    def forward(self, x):
        # Pasar por el "backbone"
        x = self.backbone(x)

        # Pasar por las capas de "head" y "aux_head"
        out = self.head(x["out"])
        aux_out = self.aux_head(x["aux"])

        return out, aux_out
