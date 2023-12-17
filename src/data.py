import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Callable, Optional
from sklearn.model_selection import train_test_split

from utils import download_dataset, DATA_DIR, read_image


class CatDogDataset(Dataset):
    """
    Cat and dog dataset.
    """

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
        """
        Return the length of the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path + ".jpg")
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, (label - 1)  # -1 because the labels start from 1 instead of 0


def create_cat_vs_dog_dataset():
    """
    Create a dataset with two classes: cat and dog.
    """
    path = os.path.join(DATA_DIR, "annotations", "list.txt")
    annotations = pd.read_csv(path, sep=" ", names=["filename", "class_id", "species", "breed_id"], skiprows=6)
    annotations.drop(columns=["breed_id", "class_id"], inplace=True)

    train, test = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations["species"])

    train.to_csv(os.path.join(DATA_DIR, "annotations", "train.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "annotations", "test.csv"), index=False)


if __name__ == "__main__":
    download_dataset()
    create_cat_vs_dog_dataset()
