import os
import uuid
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Literal
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from wandb.integration.ultralytics import add_wandb_callback

from data import CatDogDataset, AnimalSegmentationDataset
from utils import DATA_DIR, IMAGES_DIR, MODELS_DIR, get_logger
from model import CatDogClassifier, AnimalSegmentation, CatDogClassifier2, AnimalSegmentation2, DiceLoss, HeadDetection


def get_catdog_dataset() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )
    train_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "train.csv"), root_dir=IMAGES_DIR, transform=transform)
    test_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "test.csv"), root_dir=IMAGES_DIR, transform=transform)

    return train_dataset, test_dataset


def get_animalseg_dataset() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform2 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    train_dataset = AnimalSegmentationDataset(
        csv_file=os.path.join(DATA_DIR, "annotations", "train_seg.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        trimap_dir=os.path.join(DATA_DIR, "annotations", "trimaps"),
        transform=(transform, transform2),
    )
    test_dataset = AnimalSegmentationDataset(
        csv_file=os.path.join(DATA_DIR, "annotations", "test_seg.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        trimap_dir=os.path.join(DATA_DIR, "annotations", "trimaps"),
        transform=(transform, transform2),
    )

    return train_dataset, test_dataset


def load_dataset(
    dataset: Literal["catdog", "animalseg"], workers: Tuple[int, int] = (8, 4), batch_size: Tuple[int, int] = (64, 64)
) -> Tuple[DataLoader, DataLoader]:
    """
    Load the dataset and apply transformations.
    """
    dataset_map = {"catdog": get_catdog_dataset, "animalseg": get_animalseg_dataset}
    train_dataset, test_dataset = dataset_map[dataset.lower()]()

    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True, num_workers=workers[0], persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size[1], shuffle=False, num_workers=workers[1], persistent_workers=True)

    return train_loader, test_loader


def train_dogcat_classifier():
    logger = get_logger("cat_dog_classifier2.log")
    run_uuid = uuid.uuid4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogClassifier().to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    lr_adam = 1e-4
    # lr_sgd = 1e-4
    adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    # sgd = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.9)

    train_loader, test_loader = load_dataset("catdog")
    epchos = 100
    logger.info(
        f"[{run_uuid}]: Training on {len(train_loader)} samples and validating on {len(test_loader)} samples on {device} for {epchos} epochs"
    )
    wandb.init(
        project="cat-dog-classifier",
        name=f"{run_uuid}",
        config={
            "epochs": epchos,
            "batch_size": 64,
            "device": device,
            # "optimizer": ["adam", "sgd"],
            "optimizer": "adam",
            # "lr": [lr_adam, lr_sgd],
            "lr": lr_adam,
            "momentum": 0.9,
            "loss": "cross_entropy with reduction=sum",
            "dataset": "cat_dog_dataset",
            "architecture": str(model),
        },
    )
    wandb.watch(model, criterion, log_freq=10)
    min_loss = 10000000
    # opt_condition = int(0.8 * epchos)
    for epoch in range(epchos):
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        optimizer = adam  # if epoch < opt_condition else sgd
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
            pred = outputs.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(labels.view_as(pred)).sum().item()

            t.set_postfix(loss=train_loss / ((i + 1) * train_loader.batch_size), acc=train_acc / ((i + 1) * train_loader.batch_size))

        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in (
                t := tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Validation Epoch {epoch + 1}/{epchos}")
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                outputs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(labels.view_as(pred)).sum().item()

                t.set_postfix(loss=val_loss / ((i + 1) * test_loader.batch_size), acc=val_acc / ((i + 1) * test_loader.batch_size))

        tt_loss = val_loss / len(test_loader.dataset)
        tt_acc = val_acc / len(test_loader.dataset)
        t_loss = train_loss / len(train_loader.dataset)
        t_acc = train_acc / len(train_loader.dataset)

        wandb.log({"val_loss": tt_loss, "val_acc": tt_acc, "train_loss": t_loss, "train_acc": t_acc})

        if tt_loss < min_loss:
            min_loss = tt_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"cat_dog_classifier_{run_uuid}.pth"))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"cat_dog_classifier_{run_uuid}-{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"cat_dog_classifier_{run_uuid}_final.pth"))
    wandb.finish()


def train_animal_segmentation():
    logger = get_logger("animal_segmentation.log")
    run_uuid = uuid.uuid4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnimalSegmentation2().to(device)

    criterion = DiceLoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = load_dataset("animalseg")
    epchos = 20
    logger.info(
        f"[{run_uuid}]: Training on {len(train_loader)} samples and validating on {len(test_loader)} samples on {device} for {epchos} epochs"
    )
    wandb.init(
        project="animal-segmentation",
        name=f"{run_uuid}",
        config={
            "epochs": epchos,
            "batch_size": 64,
            "device": device,
            "optimizer": "adam",
            "lr": lr,
            "loss": "dice loss",
            "dataset": "animal_segmentation_dataset",
            "architecture": str(model),
        },
    )

    min_loss = 10000000
    wandb.watch(model, criterion, log_freq=10)
    for epoch in range(epchos):
        train_loss = 0.0
        model.train()

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

            t.set_postfix(loss=train_loss / ((i + 1) * train_loader.batch_size))

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in (
                t := tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Validation Epoch {epoch + 1}/{epchos}")
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                t.set_postfix(loss=val_loss / ((i + 1) * test_loader.batch_size))

        tt_loss = val_loss / len(test_loader.dataset)
        t_loss = train_loss / len(train_loader.dataset)

        wandb.log({"val_loss": tt_loss, "train_loss": t_loss})

        if tt_loss < min_loss:
            min_loss = tt_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"animal_segmentation_{run_uuid}.pth"))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"animal_segmentation_{run_uuid}-{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"animal_segmentation_{run_uuid}_final.pth"))
    wandb.finish()


def train_head_detection():
    model = HeadDetection().backbone
    uuid = uuid.uuid4()
    wandb.init(project="head-detection", name=f"{uuid}", config={"architecture": str(model)})
    add_wandb_callback(model, enable_model_checkpointing=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = model.train(data=os.path.join(DATA_DIR, "head_pos", "dataset.yaml"), epochs=50, verbose=True, device=device)
    print(results)
    wandb.finish()


if __name__ == "__main__":
    # train_dogcat_classifier()
    # train_animal_segmentation()
    train_head_detection()
