"""Create a class-balanced CIFAR-10 mini-sample index.

Repository-relative CLI paths are resolved from the repository root so the
utility behaves the same from every working directory.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_OUTPUT = (
    _REPO_ROOT / "build" / "tools" / "distill_cifar10" / "mini_samples.json"
)
_TINY_CLIP_MODEL = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
torch_device = torch.device("cpu")


def _resolve_repo_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    return resolved.resolve()


def _require_distillation_dependencies() -> None:
    torch_base_version = torch.__version__.split("+", 1)[0]
    torch_major_minor = tuple(
        int(part) for part in torch_base_version.split(".")[:2]
    )
    if torch_major_minor != (2, 8):
        raise RuntimeError(
            "CIFAR distillation requires the disposable Torch 2.8 control "
            f"profile; found torch {torch.__version__}."
        )
    try:
        import torch_kmeans  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "CIFAR distillation requires the optional contributor packages "
            "from requirements-distill.txt. They are not part of the customer "
            "QAT bundle. Install that profile into the disposable Torch 2.8 "
            "control environment; do not modify the shared Model Compiler "
            "environment."
        ) from error


def clip_embeddings(
    class_samples: np.ndarray,
    allow_download: bool,
) -> Tensor:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(
        _TINY_CLIP_MODEL,
        local_files_only=not allow_download,
    )
    processor = CLIPProcessor.from_pretrained(
        _TINY_CLIP_MODEL,
        local_files_only=not allow_download,
    )

    compute_dtype = (
        torch.float16 if torch_device.type == "cuda" else torch.float32
    )
    model.to(device=torch_device, dtype=compute_dtype).eval()
    compiled_model = torch.compile(model)

    embed_dim = 512
    sample_features = []
    print("Computing image embeddings ...")

    with torch.no_grad():
        for sample in tqdm(class_samples):
            image = Tensor(sample).to(torch_device)
            inputs = processor(
                images=image,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
            )
            inputs = {
                name: (
                    value.to(device=torch_device, dtype=compute_dtype)
                    if value.is_floating_point()
                    else value.to(torch_device)
                )
                for name, value in inputs.items()
            }
            image_features = compiled_model.get_image_features(**inputs)
            sample_features.append(image_features[0])

    # Keep four batches for the batched KMeans implementation used below.
    embed_features = torch.stack(sample_features).view((4, -1, embed_dim))
    return embed_features


def build_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    """Build deterministic CIFAR-10 iterators for index selection."""
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = CIFAR10(
        args.data_dir,
        train=True,
        download=args.allow_download,
        transform=dataset_transform,
    )
    dataset_test = CIFAR10(
        args.data_dir,
        train=False,
        download=args.allow_download,
        transform=dataset_transform,
    )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )
    return train_dataloader, test_dataloader


def get_samples(dloader: DataLoader, sample_list: np.ndarray) -> np.ndarray:
    """Collect selected samples into a contiguous array."""
    sample_set = {int(index) for index in sample_list}
    out_samples = np.zeros((len(sample_list), 3, 32, 32), dtype=np.float32)
    tail_ptr = 0
    for index, sample in enumerate(dloader):
        if index not in sample_set:
            continue
        out_samples[tail_ptr] = sample[0].cpu().numpy()
        tail_ptr += 1
    return out_samples


def check_uniqueness(values: Any) -> None:
    if isinstance(values, list):
        values = np.array(values, dtype=np.int64)
    if isinstance(values, Tensor):
        values = values.cpu().numpy()
    values = values.flatten()
    unique_count = len(set(values))
    if unique_count != values.size:
        print(f"Got {unique_count} unique indices and expected: {values.size}")


def classify_and_sort(
    dloader: DataLoader,
    n_clusters: int,
    allow_download: bool,
) -> np.ndarray:
    from torch_kmeans import KMeans

    n_classes = 10
    gt_classes = np.zeros(
        (n_classes, int(len(dloader) / n_classes)),
        dtype=np.int64,
    )
    class_ptr = np.zeros((n_classes,), dtype=np.int64)

    for index, sample in enumerate(dloader):
        class_id = sample[1].cpu().numpy()[0]
        pointer = class_ptr[class_id]
        gt_classes[class_id][pointer] = index
        class_ptr[class_id] += 1

    print(f"Got {gt_classes.shape[1]} samples for {n_classes} classes")
    check_uniqueness(gt_classes)

    compute_dtype = (
        torch.float16 if torch_device.type == "cuda" else torch.float32
    )
    kmeans = KMeans(n_clusters=n_clusters).to(torch_device)
    if compute_dtype == torch.float16:
        kmeans = kmeans.half()
    compiled_kmeans = torch.compile(kmeans)
    chosen_samples = np.zeros((n_classes, n_clusters), dtype=np.int64)

    for category in range(n_classes):
        class_samples = get_samples(dloader, sample_list=gt_classes[category])
        embeddings = clip_embeddings(
            class_samples,
            allow_download=allow_download,
        ).to(compute_dtype)

        print(
            f"Computing top-{n_clusters} feature clusters for class: "
            f"{category} ..."
        )
        category_kmeans = deepcopy(compiled_kmeans).to(torch_device)
        cluster_indices = category_kmeans.fit_predict(embeddings)
        cluster_indices = torch.flatten(cluster_indices).cpu().numpy()

        category_indices = np.array(
            [
                int(np.where(cluster_indices == cluster)[0][0])
                for cluster in range(n_clusters)
            ]
        )
        chosen_samples[category] = gt_classes[category][category_indices]

    check_uniqueness(chosen_samples)
    return chosen_samples


def write_samples(chosen_samples: dict[str, list[int]], output: Path) -> None:
    """Write the selected source-dataset indices as JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing mini samples to file: {output}")
    with output.open("w", encoding="utf-8") as stream:
        json.dump(chosen_samples, stream, indent=4)
        stream.write("\n")


def main(args: Namespace) -> None:
    _require_distillation_dependencies()
    train_dataloader, test_dataloader = build_dataloaders(args)
    subsets = {
        "train": train_dataloader,
        "test": test_dataloader,
    }
    mini_samples: dict[str, list[int]] = {
        "train": [],
        "test": [],
    }

    for name, dataloader in subsets.items():
        n_samples = int(args.sample_pct * len(dataloader))
        if n_samples < 1:
            raise ValueError(
                f"--sample-pct={args.sample_pct} selects no {name} samples"
            )
        print(f"Distilling subset: {name} to {n_samples} samples per class")
        chosen_samples = classify_and_sort(
            dataloader,
            n_clusters=n_samples,
            allow_download=args.allow_download,
        )

        for category in chosen_samples:
            mini_samples[name].extend(int(index) for index in category)
        check_uniqueness(mini_samples[name])

    write_samples(mini_samples, args.output)


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Create a class-balanced CIFAR-10 mini-sample index"
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        choices=[1],
        default=1,
        help="Loader batch size; index selection currently requires 1",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="CIFAR-10 cache (relative paths resolve from the repository root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output JSON path (relative paths resolve from the repository root)",
    )
    parser.add_argument(
        "-s",
        "--sample-pct",
        type=float,
        default=0.002,
        help="Fraction of each dataset split to retain per class",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for embedding and clustering",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit CIFAR-10 and TinyCLIP downloads when caches are missing",
    )
    args = parser.parse_args()
    if not 0.0 < args.sample_pct <= 1.0:
        parser.error("--sample-pct must be in the interval (0, 1]")
    args.data_dir = _resolve_repo_path(args.data_dir)
    args.output = _resolve_repo_path(args.output)
    return args


if __name__ == "__main__":
    run_args = get_args()
    torch_device = torch.device(run_args.device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda was requested, but CUDA is unavailable")
    print(f"Using PyTorch device: {torch_device}")
    try:
        main(run_args)
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from None
