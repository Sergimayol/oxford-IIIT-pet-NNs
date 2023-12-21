import os
import uuid
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from torchvision import transforms
from torch.utils.data import DataLoader

from data import CatDogDataset
from utils import DATA_DIR, IMAGES_DIR, get_logger
from model import AnimalSegmentation2, CatDogClassifier, AnimalSegmentation


# TODO: Change this to support all the datasets
def load_dataset() -> Tuple[DataLoader, DataLoader]:
    """
    Load the dataset and apply transformations.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=45),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        ]
    )
    train_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "train.csv"), root_dir=IMAGES_DIR, transform=transform)
    test_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "test.csv"), root_dir=IMAGES_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True)

    return train_loader, test_loader


def train_dogcat_classifier():
    logger = get_logger("cat_dog_classifier.log")
    run_uuid = uuid.uuid4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    adam = torch.optim.Adam(model.parameters(), lr=1e-4)
    sgd = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    train_loader, test_loader = load_dataset()
    epchos = 100
    logger.info(
        f"[{run_uuid}]: Training on {len(train_loader)} samples and validating on {len(test_loader)} samples on {device} for {epchos} epochs"
    )

    opt_condition = int(0.8 * epchos)
    for epoch in range(epchos):
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        optimizer = adam if epoch < opt_condition else sgd
        for i, (inputs, labels) in (
            t := tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{epchos}")
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            outputs = torch.softmax(outputs, dim=1)
            train_acc += (outputs.argmax(1) == labels).sum().item()

            t.set_postfix(loss=train_loss / ((i + 1) * train_loader.batch_size), acc=train_acc / ((i + 1) * train_loader.batch_size))

        logger.info(
            f"Training Loss (Epoch {epoch + 1}): {train_loss / len(train_loader)}, Training Accuracy: {train_acc / len(train_loader)}"
        )

        val_loss = 0.0
        val_acc = 0.0
        model.eval()

        for i, (inputs, labels) in (
            t := tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Validation Epoch {epoch + 1}/{epchos}")
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            outputs = torch.softmax(outputs, dim=1)
            val_acc += (outputs.argmax(1) == labels).sum().item()

            t.set_postfix(loss=val_loss / ((i + 1) * test_loader.batch_size), acc=val_acc / ((i + 1) * test_loader.batch_size))

        logger.info(f"Val Loss (Epoch {epoch + 1}): {train_loss / len(test_loader)}, Val Accuracy: {val_acc / len(test_loader)}")

        if (epoch + 1) % 10 == 0:
            save_file_name = os.path.join(DATA_DIR, "models", f"cdc-{run_uuid}-{epoch+1}.pth")
            torch.save(model.state_dict(), save_file_name)


if __name__ == "__main__":
    logger = get_logger("animal_segmentation.log")
    run_uuid = uuid.uuid4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnimalSegmentation2().to(device)
    print(model)
    out = model(torch.randn(1, 3, 256, 256).to(device))
    from torchinfo import summary

    summary(model, input_size=(1, 3, 256, 256))
