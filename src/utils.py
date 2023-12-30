import tarfile
import urllib.request
from tqdm import tqdm
from PIL import Image
from typing import Callable
from torchinfo import summary
import os, time, contextlib, cProfile, pstats


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = "D:\\models"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DATASET_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
DATASET_GROUND_TRUTH_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"


def _tqdm_hook(t: tqdm) -> Callable[[int, int, int], None]:
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


def download_dataset(url: str = DATASET_URL, ground_truth_url: str = DATASET_GROUND_TRUTH_URL, data_dir: str = DATA_DIR, force: bool = False):
    """Download data from the web and save it to the data directory."""
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


def read_image(path: str, mode: str = "RGB") -> Image:
    """Read an image from the disk."""
    return Image.open(path).convert(mode)


def get_logger(file_name: str = "model.log"):
    """Create a logger."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="./logs/" + file_name,
        filemode="a",
    )
    return logging.getLogger(__name__)


def write_to_file(file_name: str, data: str, mode: str = "a"):
    """Write data to a file."""
    with open(file_name, mode) as f:
        f.write(data)


def create_dir(dir_name: str):
    """Create a directory."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_multiple_dirs(dir_names: list):
    """Create multiple directories."""
    for dir_name in dir_names:
        create_dir(dir_name)


def print_model_summary(model, input_size, verbose: bool = True):
    """Print model summary."""
    if verbose:
        print(model)
        summary(model, input_size=input_size)


class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True):
        self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled

    def __enter__(self):
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled:
            print(f"{self.prefix}{self.et*1e-6:.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))


class Profiling(contextlib.ContextDecorator):
    def __init__(self, enabled=True, sort="cumtime", frac=0.2):
        self.enabled, self.sort, self.frac = enabled, sort, frac

    def __enter__(self):
        self.pr = cProfile.Profile(timer=lambda: int(time.time() * 1e9), timeunit=1e-6)
        if self.enabled:
            self.pr.enable()

    def __exit__(self, *exc):
        if self.enabled:
            self.pr.disable()
            pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort).print_stats(self.frac)
