import os
import uuid
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Tuple, Literal, Dict, Callable

from data import CatDogDataset, AnimalSegmentationDataset, RaceDataset
from utils import DATA_DIR, IMAGES_DIR, MODELS_DIR, print_model_summary
from model import (
    CatDogClassifier,
    AnimalSegmentation,
    CatDogClassifierV2,
    AnimalSegmentationPretained,
    AnimalSegmentationPretained2,
    DiceLoss,
    HeadDetection,
    RaceClassifier,
)


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
    transform2 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    train_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "train.csv"), root_dir=IMAGES_DIR, transform=transform)
    test_dataset = CatDogDataset(csv_file=os.path.join(DATA_DIR, "annotations", "test.csv"), root_dir=IMAGES_DIR, transform=transform2)

    return train_dataset, test_dataset


def get_animalseg_dataset() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


def get_race_dataset() -> Tuple[Dataset, Dataset]:
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
    transform2 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    train_dataset = RaceDataset(csv_file=os.path.join(DATA_DIR, "annotations", "train_race.csv"), root_dir=IMAGES_DIR, transform=transform)
    test_dataset = RaceDataset(csv_file=os.path.join(DATA_DIR, "annotations", "test_race.csv"), root_dir=IMAGES_DIR, transform=transform2)

    return train_dataset, test_dataset


def load_dataset(
    dataset: Literal["catdog", "animalseg", "race"], workers: Tuple[int, int] = (8, 4), batch_size: Tuple[int, int] = (64, 64)
) -> Tuple[DataLoader, DataLoader]:
    """Load the dataset and apply transformations."""
    if dataset not in ["catdog", "animalseg", "race"]:
        return None, None
    dataset_map = {"catdog": get_catdog_dataset, "animalseg": get_animalseg_dataset, "race": get_race_dataset}
    train_dataset, test_dataset = dataset_map[dataset.lower()]()

    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True, num_workers=workers[0], persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size[1], shuffle=False, num_workers=workers[1], persistent_workers=True)

    return train_loader, test_loader


def get_device(device: str = "auto") -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    device: str,
    calc_acc: bool = True,
    verbose: bool = False,
) -> Tuple[float, float]:
    train_loss, train_acc = 0.0, 0.0
    model.train()
    t = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{epochs}", disable=not verbose)
    for i, (inputs, labels) in t:
        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        if calc_acc:
            outputs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(labels.view_as(pred)).sum().item()
            postfix_data = {"loss": train_loss / ((i + 1) * train_loader.batch_size), "acc": train_acc / ((i + 1) * train_loader.batch_size)}
        else:
            postfix_data = {"loss": train_loss / ((i + 1) * train_loader.batch_size)}

        t.set_postfix(**postfix_data)

    t_loss = train_loss / len(train_loader.dataset)
    t_acc = train_acc / len(train_loader.dataset)

    return t_loss, t_acc


def validate(
    model: nn.Module, test_loader: DataLoader, criterion: nn.Module, epoch: int, epochs: int, device: str, calc_acc: bool = True, verbose: bool = False
) -> Tuple[float, float]:
    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        t = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Validation Epoch {epoch + 1}/{epochs}", disable=not verbose)
        for i, (inputs, labels) in t:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            if calc_acc:
                outputs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(labels.view_as(pred)).sum().item()
                postfix_data = {"loss": val_loss / ((i + 1) * test_loader.batch_size), "acc": val_acc / ((i + 1) * test_loader.batch_size)}
            else:
                postfix_data = {"loss": val_loss / ((i + 1) * test_loader.batch_size)}

            t.set_postfix(**postfix_data)

    tt_loss = val_loss / len(test_loader.dataset)
    tt_acc = val_acc / len(test_loader.dataset)

    return tt_loss, tt_acc


def train_dogcat_classifier(
    workers: Tuple[int, int] = (8, 4),
    batch_size: Tuple[int, int] = (64, 64),
    epochs: int = 100,
    lr: float = 1e-4,
    optimizer: Literal["adam", "sgd"] = "adam",
    momentum: float = 0.9,
    device: str = "auto",
    verbose: bool = False,
    model_version: Literal["v1", "v2"] = "v1",
    **kwargs,
):
    run_uuid = uuid.uuid4()
    device = get_device(device)
    model = CatDogClassifier() if model_version == "v1" else CatDogClassifierV2()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    opt = torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_loader, test_loader = load_dataset("catdog", workers=workers, batch_size=batch_size)

    if USE_WANDB:
        wandb.init(
            project="cat-dog-classifier",
            name=f"{run_uuid}",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "device": device,
                "optimizer": str(opt),
                "lr": lr,
                "loss": "cross_entropy with reduction=sum",
                "dataset": "cat_dog_dataset",
                "architecture": str(model),
            },
        )

    print_model_summary(model, input_size=(1, 3, 256, 256), verbose=verbose)

    min_loss = 10000000
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, opt, epoch, epochs, device, verbose=verbose)
        val_loss, val_acc = validate(model, test_loader, criterion, epoch, epochs, device, verbose=verbose)

        if USE_WANDB:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc, "train_loss": train_loss, "train_acc": train_acc})

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"cat_dog_classifier_{run_uuid}.pth"))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"cat_dog_classifier_{run_uuid}_final.pth"))
    if USE_WANDB:
        wandb.finish()


def train_animal_segmentation(
    workers: Tuple[int, int] = (8, 4),
    batch_size: Tuple[int, int] = (64, 64),
    epochs: int = 100,
    lr: float = 1e-4,
    optimizer: Literal["adam", "sgd"] = "adam",
    momentum: float = 0.9,
    device: str = "auto",
    model_version: Literal["v1", "v2"] = "v1",
    verbose: bool = False,
    **kwargs,
):
    run_uuid = uuid.uuid4()
    device = get_device(device)
    model = AnimalSegmentationPretained() if model_version == "v1" else AnimalSegmentationPretained2()
    model = model.to(device)

    criterion = DiceLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_loader, test_loader = load_dataset("animalseg", workers=workers, batch_size=batch_size)
    if USE_WANDB:
        wandb.init(
            project="animal-segmentation",
            name=f"{run_uuid}",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "device": device,
                "optimizer": str(opt),
                "lr": lr,
                "loss": "dice loss",
                "dataset": "animal_segmentation_dataset",
                "architecture": str(model),
            },
        )

    print_model_summary(model, input_size=(1, 3, 256, 256), verbose=verbose)

    min_loss = 10000000
    for epoch in range(epochs):
        train_loss, _ = train(model, train_loader, criterion, opt, epoch, epochs, device, verbose=verbose, calc_acc=False)
        val_loss, _ = validate(model, test_loader, criterion, epoch, epochs, device, verbose=verbose, calc_acc=False)

        if USE_WANDB:
            wandb.log({"val_loss": val_loss, "train_loss": train_loss})

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"animal_segmentation_{run_uuid}.pth"))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"animal_segmentation_{run_uuid}_final.pth"))
    if USE_WANDB:
        wandb.finish()


def train_head_detection(epochs: int = 50, device: str = "auto", verbose: bool = False, **kwargs):
    model = HeadDetection().backbone
    run_uuid = uuid.uuid4()
    wandb.init(project="head-detection", name=f"{run_uuid}", config={"architecture": str(model)})
    device = get_device(device)
    results = model.train(data=os.path.join(DATA_DIR, "head_pos", "dataset.yaml"), epochs=epochs, verbose=verbose, device=device)
    if verbose:
        print(results)
    wandb.finish()


def train_race_classifier(
    workers: Tuple[int, int] = (8, 4),
    batch_size: Tuple[int, int] = (64, 64),
    epochs: int = 100,
    lr: float = 1e-4,
    optimizer: Literal["adam", "sgd"] = "adam",
    momentum: float = 0.9,
    device: str = "auto",
    verbose: bool = False,
    **kwargs,
):
    run_uuid = uuid.uuid4()
    device = get_device(device)
    model = RaceClassifier().to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    opt = torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_loader, test_loader = load_dataset("race", workers=workers, batch_size=batch_size)

    if USE_WANDB:
        wandb.init(
            project="race-classifier",
            name=f"{run_uuid}",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "device": device,
                "optimizer": str(opt),
                "lr": lr,
                "loss": "cross_entropy with reduction=sum",
                "dataset": "race dataset",
                "architecture": str(model),
            },
        )

    print_model_summary(model, input_size=(1, 3, 256, 256), verbose=verbose)

    min_loss = 10000000
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, opt, epoch, epochs, device, verbose=verbose)
        val_loss, val_acc = validate(model, test_loader, criterion, epoch, epochs, device, verbose=verbose)

        if USE_WANDB:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc, "train_loss": train_loss, "train_acc": train_acc})

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"race_classifier_{run_uuid}.pth"))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"race_classifier_{run_uuid}_final.pth"))
    if USE_WANDB:
        wandb.finish()


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--model", "-m", type=str, help="Model to train", choices=["catdog", "animalseg", "headpos", "race"], required=True)
    parser.add_argument("--workers", "-w", type=int, nargs=2, default=(8, 4), help="Number of workers for train and test, respectively")
    parser.add_argument("--batch-size", "-b", type=int, nargs=2, default=(64, 64), help="Batch size for train and test, respectively")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", "-l", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--momentum", "-mo", type=float, default=0.9, help="Momentum")
    parser.add_argument("--optimizer", "-o", type=str, default="adam", help="Optimizer", choices=["adam", "sgd"])
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device to train on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    parser.add_argument("--wandb", "-wdb", action="store_true", help="Log to wandb")
    parser.add_argument("--model-version", "-mv", type=str, default="v1", help="Model version", choices=["v1", "v2"])
    return parser.parse_args()


USE_WANDB = False

if __name__ == "__main__":
    args = __parse_args()
    USE_WANDB = args.wandb
    if USE_WANDB:
        import wandb
    model_train_map: Dict[str, Callable] = {
        "catdog": train_dogcat_classifier,
        "race": train_race_classifier,
        "headpos": train_head_detection,
        "animalseg": train_animal_segmentation,
    }
    model_train_map[args.model.lower()](**vars(args))
