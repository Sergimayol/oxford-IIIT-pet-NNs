import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

from data import CatDogDataset
from model import CatDogClassifier
from utils import DATA_DIR, IMAGES_DIR, get_logger


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
        ]
    )
    train_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "train.csv"), root_dir=IMAGES_DIR, transform=transform)
    test_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "test.csv"), root_dir=IMAGES_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == "__main__":
    logger = get_logger("cat_dog_classifier.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(DATA_DIR, "models", "cat_dog_classifier-20231217152013-25.pth")
    model = CatDogClassifier().to(device)
    model.load_state_dict(torch.load(path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader, test_loader = load_dataset()

    epchos = 50
    for epoch in range(epchos):
        epoch += 25
        train_loss = 0.0
        train_acc = 0.0

        val_loss = 0.0
        val_acc = 0.0

        model.train()

        for i, (inputs, labels) in (
            t := tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Epoch {}/{}".format(epoch + 1, epchos + 25))
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
            f"Training Loss (Epoch {epoch + 1})): {train_loss / len(train_loader)}, Training Accuracy: {train_acc / len(train_loader)}"
        )

        model.eval()
        for i, (inputs, labels) in (
            t := tqdm(enumerate(test_loader), total=len(test_loader), desc="Validation Epoch {}/{}".format(epoch + 1, epchos + 25))
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            outputs = torch.softmax(outputs, dim=1)
            val_acc += (outputs.argmax(1) == labels).sum().item()

            t.set_postfix(loss=val_loss / ((i + 1) * test_loader.batch_size), acc=val_acc / ((i + 1) * test_loader.batch_size))

        logger.info(f"Val Loss (Epoch {epoch + 1})): {train_loss / len(train_loader)}, Val Accuracy: {val_acc / len(train_loader)}")

        save_file_name = os.path.join(DATA_DIR, "models", f"cat_dog_classifier-{datetime.now().strftime('%Y%m%d%H%M%S')}-{epoch+1}.pth")
        torch.save(model.state_dict(), save_file_name)
