import os
import tarfile
import urllib.request
from tqdm import tqdm
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = "D:\\models"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DATASET_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
DATASET_GROUND_TRUTH_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"


def _tqdm_hook(t: tqdm):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks just transferred [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download_dataset(
    url: str = DATASET_URL, ground_truth_url: str = DATASET_GROUND_TRUTH_URL, data_dir: str = DATA_DIR, force: bool = False
):
    """
    Download data from the web and save it to the data directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = url.split("/")[-1]
    ground_truth_filename = ground_truth_url.split("/")[-1]

    if not os.path.exists(os.path.join(data_dir, filename)) or force:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=os.path.join(data_dir, filename), reporthook=_tqdm_hook(t))

    if not os.path.exists(os.path.join(data_dir, ground_truth_filename)) or force:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=ground_truth_filename) as t:
            urllib.request.urlretrieve(ground_truth_url, filename=os.path.join(data_dir, ground_truth_filename), reporthook=_tqdm_hook(t))

    if not os.path.exists(os.path.join(data_dir, "images")) or force:
        with tarfile.open(os.path.join(data_dir, filename)) as tar:
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc="Extracting " + filename):
                tar.extract(member, path=data_dir)

    if not os.path.exists(os.path.join(data_dir, "annotations")) or force:
        with tarfile.open(os.path.join(data_dir, ground_truth_filename)) as tar:
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc="Extracting " + ground_truth_filename):
                tar.extract(member, path=data_dir)


def read_image(path: str, mode: str = "RGB"):
    """
    Read an image from the disk.
    """
    return Image.open(path).convert(mode)


def get_logger(file_name: str = "model.log"):
    """
    Create a logger.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="./logs/" + file_name,
        filemode="a",
    )
    logger = logging.getLogger(__name__)
    return logger
