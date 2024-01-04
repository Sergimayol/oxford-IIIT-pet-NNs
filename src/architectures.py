import torch
from utils import print_model_summary
from model import CatDogClassifier, CatDogClassifierV2, RaceClassifier, HeadDetection, AnimalSegmentationPretained, AnimalSegmentationPretained2

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models_input = [
        (CatDogClassifier().to(device), (1, 3, 256, 256)),
        (CatDogClassifierV2().to(device), (1, 3, 256, 256)),
        (RaceClassifier().to(device), (1, 3, 256, 256)),
        (HeadDetection().to(device), None),
        (AnimalSegmentationPretained().to(device), (1, 3, 256, 256)),
        (AnimalSegmentationPretained2().to(device), (1, 3, 256, 256)),
    ]

    for model, input in models_input:
        print_model_summary(model, input)
        print("\n")
