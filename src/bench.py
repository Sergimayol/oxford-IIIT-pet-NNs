"""Benchmarks for the project. Run with python -m src.bench --help for more info.
Examples: 
    - python ./src/bench.py -b race -mp D:/models/race_classifier_e3136a0b-77a1-4335-b833-55ed999a1a52.pth -v
    - python ./src/bench.py -mv v1 -mp D:/models/cat_dog_classifier_c33726a6-07cb-43e4-86ec-43451067f06f.pth -v
"""
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader
from typing import Dict, Callable, Literal
import os, cProfile, argparse, contextlib, time, pstats, warnings

from model import CatDogClassifier, CatDogClassifierV2, RaceClassifier
from train import load_dataset
from utils import read_image, DATA_DIR


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


def catdog_bench(
    test_data: DataLoader,
    model_version: Literal["v1", "v2"] = "v1",
    device: str = "auto",
    model_path: str = None,
    verbose: bool = False,
    profile: bool = False,
    **kwargs,
):
    """Benchmark for the catdog classifier. Returns the accuracy and the inference time. Runs a benchmark on home made model vs ultralytics model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    yolo = YOLO("yolov8x.pt").to(device)
    model = CatDogClassifier() if model_version == "v1" else CatDogClassifierV2()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    acc, acc_yolo = 0.0, 0.0
    with Profiling(enabled=profile):
        st = time.perf_counter_ns()
        for img, label in tqdm(test_data, desc="CatDog benchmark", total=len(test_data), disable=not verbose):
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                outputs = model(img)
                outputs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1, keepdim=True)
                acc += pred.eq(label.view_as(pred)).sum().item()
        et = time.perf_counter_ns() - st
        model_time = et * 1e-9

    df = pd.read_csv(os.path.join(DATA_DIR, "annotations", "test.csv"))
    imgs = df["filename"].apply(lambda x: read_image(os.path.join(DATA_DIR, "images", f"{x}.jpg")))
    labels = df["species"]

    with Profiling(enabled=profile):
        st = time.perf_counter_ns()
        for img, label in tqdm(zip(imgs, labels), desc="YOLO benchmark", total=len(imgs), disable=not verbose):
            with torch.no_grad():
                outputs = yolo.predict(img, verbose=False)
                result = outputs[0]
                if len(result) == 0:
                    continue
                for box in result.boxes:
                    class_id = result.names[box.cls[0].item()]
                    if (class_id == "cat" and label == 0) or (class_id == "dog" and label == 1):
                        acc_yolo += 1
                        break
        et = time.perf_counter_ns() - st
        yolo_time = et * 1e-9
    accuracy = acc / len(test_data.dataset)
    accuracy_yolo = acc_yolo / len(test_data.dataset)
    print(f"[INFO] Accuracy: {accuracy*100:.2f}%, Accuracy YOLO: {accuracy_yolo*100:.2f}% | Model time: {model_time:.2f}s, YOLO time: {yolo_time:.2f}s")


def race_bench(test_data: DataLoader, device: str = "auto", model_path: str = None, verbose: bool = False, profile: bool = False, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model = RaceClassifier()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    acc = 0.0
    with Profiling(enabled=profile):
        st = time.perf_counter_ns()
        for img, label in tqdm(test_data, desc="CatDog benchmark", total=len(test_data), disable=not verbose):
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                outputs = model(img)
                outputs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1, keepdim=True)
                acc += pred.eq(label.view_as(pred)).sum().item()
        et = time.perf_counter_ns() - st
        model_time = et * 1e-9
    accuracy = acc / len(test_data.dataset)
    print(f"[INFO] Accuracy: {accuracy*100:.2f}% | Model time: {model_time:.2f}s")


def __argparse():
    parser = argparse.ArgumentParser(description="Benchmarks for the project")
    parser.add_argument("--bench", "-b", type=str, default="catdog", help="Benchmarks to run", choices=["catdog", "race"])
    parser.add_argument("--model-version", "-mv", type=str, default="v1", help="Model version", choices=["v1", "v2"])
    parser.add_argument("--model-path", "-mp", type=str, required=True, help="Path to the model (pth file)")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device to bench on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    parser.add_argument("--profile", "-p", action="store_true", help="Profile mode")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = __argparse()
    bench_map: Dict[str, Callable] = {"catdog": catdog_bench, "race": race_bench}
    bench: str = args.bench
    print(f"[INFO] Running benchmark {bench}")
    with Timing(prefix="[INFO] Total time: "):
        _, test_data = load_dataset(bench, workers=(1, 1), batch_size=(1, 1))
        bench_map[bench](test_data, **vars(args))
