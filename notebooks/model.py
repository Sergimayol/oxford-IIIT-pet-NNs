"""Copy of model.py from src folder. Used for sample.ipynb notebook."""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ultralytics import YOLO
from collections import OrderedDict


class AlexNet(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 1000, freeze_backbone: bool = True):
        super(AlexNet, self).__init__()
        model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=pretrained)
        self.backbone: nn.Sequential = model.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier: nn.Sequential = model.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)
        if pretrained and freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x) -> Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CatDogClassifier(nn.Module):
    """Cat and dog classifier. Base architecture from AlexNet."""

    def __init__(self, base: Optional[AlexNet] = None, num_classes: int = 2):
        super(CatDogClassifier, self).__init__()
        if base is None:
            base = AlexNet(pretrained=False, num_classes=num_classes)
        self.features = base.backbone
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CatDogClassifierV2(nn.Module):
    """Cat and dog classifier V2."""

    def __init__(self, num_classes: int = 2):
        super(CatDogClassifierV2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),  # 256x256x3 -> 256x256x10
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 256x256x10 -> 128x128x10
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),  # 128x128x10 -> 128x128x20
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 128x128x20 -> 64x64x20
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1, padding=1),  # 64x64x20 -> 64x64x30
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, stride=1, padding=1),  # 64x64x30 -> 64x64x30
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 64x64x30 -> 32x32x30
            nn.Conv2d(in_channels=30, out_channels=40, kernel_size=3, stride=1, padding=1),  # 32x32x30 -> 32x32x40
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1),  # 32x32x40 -> 32x32x40
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 32x32x40 -> 16x16x40
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 * 40, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RaceClassifier(nn.Module):
    def __init__(self, base: Optional[AlexNet] = None, num_classes: int = 37):
        super(RaceClassifier, self).__init__()
        if base is None:
            base = AlexNet(pretrained=True, num_classes=num_classes)
        self.backbone = base.backbone
        self.avgpool = base.avgpool
        self.classifier = base.classifier

    def forward(self, x) -> Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class HeadDetection(nn.Module):
    """Head detection model. Using pretrained model YOLOv8"""

    def __init__(self, model_path: str = "yolov8n.pt"):
        super(HeadDetection, self).__init__()
        self.backbone = YOLO(model_path)

    def forward(self, x) -> Tensor:
        return self.backbone.predict(x)


class AnimalSegmentationPretained(nn.Module):
    """Animal segmentation model. Using pretrained model from mateuszbuda/brain-segmentation-pytorch"""

    def __init__(self):
        super(AnimalSegmentationPretained, self).__init__()
        self.backbone = torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet", in_channels=3, out_channels=1, init_features=32, pretrained=True)

    def forward(self, x) -> Tensor:
        x = self.backbone(x)
        return torch.sigmoid(x)


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels, out_size):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
            nn.Upsample(size=out_size, mode="bilinear", align_corners=True),
        ]
        super(FCNHead, self).__init__(*layers)


class AnimalSegmentationPretained2(nn.Module):
    """Animal segmentation model. Using pretrained model from mateuszbuda/brain-segmentation-pytorch"""

    def __init__(self):
        super(AnimalSegmentationPretained2, self).__init__()
        self.backbone: nn.Sequential = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=True).backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = FCNHead(2048, 1, (256, 256))

    def forward(self, x) -> Tensor:
        x = self.backbone(x)["out"]
        x = self.classifier(x)
        return torch.sigmoid(x)


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks. https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py"""

    def __init__(self, smooth: float = 0.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert y_pred.size() == y_true.size(), f"Predicted and ground truth sizes do not match {y_pred.size()} != {y_true.size()}"
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1.0 - dsc


class AnimalSegmentation(nn.Module):
    """Animal segmentation model. Architecture from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py"""

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(AnimalSegmentation, self).__init__()
        features = init_features
        self.encoder1 = AnimalSegmentation._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = AnimalSegmentation._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = AnimalSegmentation._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = AnimalSegmentation._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = AnimalSegmentation._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = AnimalSegmentation._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = AnimalSegmentation._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = AnimalSegmentation._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = AnimalSegmentation._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
