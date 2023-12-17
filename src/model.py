import torch
import torch.nn as nn


class CatDogClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CatDogClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        # Maybe reduce the number of neurons in the classifier
        # self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(256 * 16 * 16, 1024),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(1024, 512),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(512, num_classes),
        #    nn.Softmax(dim=1),
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
            #nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RaceClassifier(nn.Module):
    pass


class HeadDetection(nn.Module):
    pass


class AnimalSegmentation(nn.Module):
    pass
