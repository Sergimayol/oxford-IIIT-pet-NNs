import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
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


class CatDogClassifier2(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(CatDogClassifier2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)  # 256x256x3 -> 256x256x10
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 256x256x10 -> 128x128x10
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)  # 128x128x10 -> 128x128x20
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 128x128x20 -> 64x64x20
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1, padding=1)  # 64x64x20 -> 64x64x30
        self.conv4 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, stride=1, padding=1)  # 64x64x30 -> 64x64x30
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 64x64x30 -> 32x32x30
        self.conv5 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=3, stride=1, padding=1)  # 32x32x30 -> 32x32x40
        self.conv6 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)  # 32x32x40 -> 32x32x40
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 32x32x40 -> 16x16x40
        self.fc1 = nn.Linear(16 * 16 * 40, 1024)  # 16x16x40 -> 1024
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)  # 1024 -> 512
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)  # 512 -> 256
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, num_classes)  # 256 -> num_classes

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp3(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.mp4(x)
        x = x.view(-1, 16 * 16 * 40)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
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
