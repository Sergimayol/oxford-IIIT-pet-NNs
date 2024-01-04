import os
import argparse
import pandas as pd
from torch import Tensor
from tqdm import tqdm
from shutil import copy
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, Dict
from sklearn.model_selection import train_test_split

from utils import download_dataset, DATA_DIR, read_image, write_to_file, create_dir, create_multiple_dirs


class CatDogDataset(Dataset):
    """Cat and dog dataset."""

    def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path + ".jpg")
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label


class AnimalSegmentationDataset(Dataset):
    """Animal segmentation dataset."""

    def __init__(self, csv_file: str, root_dir: str, trimap_dir: str, transform: Tuple[Optional[Callable], Optional[Callable]] = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            trimap_dir (string): Directory with all the trimap images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.trimap_dir = trimap_dir
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        trimap_path = os.path.join(self.trimap_dir, self.annotations.iloc[idx, 0])
        image, label = read_image(img_path + ".jpg"), read_image(trimap_path + ".png", mode="L")
        if self.transform:
            assert len(self.transform) == 2, "The transform must be a tuple of two elements"
            image = self.transform[0](image)
            label: Tensor = (self.transform[1](label) * 255).long()
            label[label == 1] = 1
            label[label == 2] = 0
            label[label == 3] = 1
            label = label.float() / 255
        return image, label


class RaceDataset(Dataset):
    """Cat dog race dataset."""

    def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path + ".jpg")
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # label: (0, 36)
        return image, label - 1  # -1 because the class_id starts at 1


def create_cat_vs_dog_dataset():
    print("[INFO]: Creating cat vs dog dataset")
    path = os.path.join(DATA_DIR, "annotations", "list.txt")
    annotations = pd.read_csv(path, sep=" ", names=["filename", "class_id", "species", "breed_id"], skiprows=6)
    annotations.drop(columns=["breed_id", "class_id"], inplace=True)

    train, test = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations["species"])

    train["species"] = train["species"].apply(lambda x: 0 if x == 1 else 1)
    test["species"] = test["species"].apply(lambda x: 0 if x == 1 else 1)

    train.to_csv(os.path.join(DATA_DIR, "annotations", "train.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "annotations", "test.csv"), index=False)
    print(f"[INFO]: Cat vs dog dataset created in {os.path.join(DATA_DIR, 'annotations')}")


def create_animal_segmentation_dataset():
    print("[INFO]: Creating animal segmentation dataset")
    path = os.path.join(DATA_DIR, "annotations", "list.txt")
    annotations = pd.read_csv(path, sep=" ", names=["filename", "class_id", "species", "breed_id"], skiprows=6)
    annotations.drop(columns=["class_id", "breed_id"], inplace=True)
    train, test = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations["species"])

    train.to_csv(os.path.join(DATA_DIR, "annotations", "train_seg.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "annotations", "test_seg.csv"), index=False)
    print(f"[INFO]: Animal segmentation dataset created in {os.path.join(DATA_DIR, 'annotations')}")


def create_head_pos_dataset():
    print("[INFO]: Creating head position dataset")
    path = os.path.join(DATA_DIR, "annotations", "xmls")
    annotations = []

    final_path = os.path.join(DATA_DIR, "head_pos")
    create_dir(final_path)

    dataset_yaml_path = os.path.join(DATA_DIR, "head_pos", "dataset.yaml")
    write_to_file(dataset_yaml_path, "train: ./train\nval: ./test\nnames:\n 0: head", mode="w")

    for file in tqdm(os.listdir(path)):
        if file.endswith(".xml"):
            tree = ET.parse(os.path.join(path, file))
            root = tree.getroot()
            filename = root.find("filename").text
            bbox = root.find("object").find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            width = float(root.find("size").find("width").text)
            height = float(root.find("size").find("height").text)

            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            width = (xmax - xmin) / width
            height = (ymax - ymin) / height
            filename = filename.split(".")[0]
            content = f"0 {x_center} {y_center} {width} {height}"
            annotations.append([content, 0, filename + ".txt"])

    annotations = pd.DataFrame(annotations, columns=["content", "label", "filename"])
    train, test = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations["label"])

    create_multiple_dirs(
        [
            os.path.join(final_path, "train", "labels"),
            os.path.join(final_path, "train", "images"),
            os.path.join(final_path, "test", "labels"),
            os.path.join(final_path, "test", "images"),
        ]
    )

    img_path = lambda x: os.path.join(DATA_DIR, "images", x["filename"].split(".")[0] + ".jpg")

    train.apply(lambda x: copy(img_path(x), os.path.join(final_path, "train", "images")), axis=1)
    test.apply(lambda x: copy(img_path(x), os.path.join(final_path, "test", "images")), axis=1)

    train.apply(lambda x: write_to_file(os.path.join(final_path, "train", "labels", x["filename"]), x["content"], "w"), axis=1)
    test.apply(lambda x: write_to_file(os.path.join(final_path, "test", "labels", x["filename"]), x["content"], "w"), axis=1)

    print(f"[INFO]: Head position dataset created in {os.path.join(DATA_DIR, 'head_pos')}")


def create_race_dataset():
    # class_id 1-37 -> class_*.jpg
    print("[INFO]: Creating cat dog race dataset")
    path = os.path.join(DATA_DIR, "annotations", "list.txt")
    annotations = pd.read_csv(path, sep=" ", names=["filename", "class_id", "species", "breed_id"], skiprows=6)
    annotations.drop(columns=["species", "breed_id"], inplace=True)

    train, test = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations["class_id"])

    # breed_name = filename without_extension and numbers
    get_non_numbers = lambda x: "".join([c for c in x if not c.isdigit()])
    breed_name = lambda x: get_non_numbers(x["filename"].split(".")[0])[:-1]
    train["breed_name"] = train.apply(breed_name, axis=1)
    test["breed_name"] = test.apply(breed_name, axis=1)

    train.to_csv(os.path.join(DATA_DIR, "annotations", "train_race.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "annotations", "test_race.csv"), index=False)
    print(f"[INFO]: Cat dog race dataset created in {os.path.join(DATA_DIR, 'annotations')}")


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="catdog",
        help="Dataset to create from the downloaded files",
        choices=["catdog", "animal_segmentation", "head_pos", "race"],
    )
    parser.add_argument("--all", "-a", action="store_true", help="Create all the dataset from the downloaded files", default=False)
    parser.add_argument("--force_download", "-f", action="store_true", help="Force download dataset", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = __parse_args()
    download_dataset(force=args.force_download)
    dataset_map: Dict[str, Callable] = {
        "catdog": create_cat_vs_dog_dataset,
        "animal_segmentation": create_animal_segmentation_dataset,
        "head_pos": create_head_pos_dataset,
        "race": create_race_dataset,
    }
    if args.all:
        for dataset in dataset_map.values():
            dataset()
    else:
        dataset_map[args.dataset.lower()]()
