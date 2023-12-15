import torch
from utils import download_dataset

if __name__ == "__main__":
    download_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
