
import os
import logging
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
from copy import deepcopy

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import DenseNet
# from torchsummary import summary

from torch import optim, nn, utils, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# For embedding processing
from transformers import CLIPProcessor, CLIPModel
from torch_kmeans import KMeans

# Distill the CIFAR10 samples down to 1000 samples.

torch_device = None


def clip_embeddings(class_samples: np.ndarray) -> np.ndarray:
    model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")

    model.to(torch_device)
    c_model = torch.compile(model)

    embed_dim = 512
    sample_features = []
    print(f"Computing image embeddings ...")

    with torch.no_grad(): # torch.autocast(device_type="mps", dtype=torch.float16):
        for i in tqdm(range(len(class_samples))):
            image = Tensor(class_samples[i]).to(torch_device)
            inputs = processor(images=image, return_tensors="pt", padding=True, do_rescale=False)
            for k, v in inputs.items():
                inputs[k] = v.half().to(torch_device)

            image_features = c_model.get_image_features(**inputs)
            
            sample_features.append(image_features[0])
    
    # Choose batch=4 to improve downstream runtime
    embed_features = torch.stack(sample_features).view((4, -1, embed_dim))
    # Save memory
    del sample_features
    return embed_features


def build_dataloaders(args: Namespace):
    """ Build the dataset iterators
    """
    # We need absolutely minimal transforms here because the CLIP embedding needs unprocessed 
    # image data.
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_train = CIFAR10(args.data, train=True, download=True, transform=test_transforms)
    dataset_test = CIFAR10(args.data, train=False, download=True, transform=test_transforms)

    workers = 0
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch, shuffle=False, num_workers=workers) # persistent_workers=True)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch, shuffle=False, num_workers=workers)  #, persistent_workers=True)
    return train_dataloader, test_dataloader


def get_samples(dloader: DataLoader, sample_list: np.ndarray) -> np.ndarray:
    # Collect given samples into a single array.
    sample_set = set([int(x) for x in sample_list])
    out_samples = np.zeros((len(sample_list), 3, 32, 32), dtype=np.float32)
    tail_ptr = 0
    for i, s in enumerate(dloader):
        if i not in sample_set:
            continue
        out_samples[tail_ptr] = s[0].cpu().numpy()
        tail_ptr += 1
    return out_samples


def check_uniqueness(x):
    if isinstance(x, list):
        x = np.array(x, dtype=np.int64)
    if isinstance(x, Tensor):
        x = x.cpu().numpy()
    x = x.flatten()
    uniques = set(x)
    lu = len(uniques)
    if lu != x.size:
        print(f"Got {lu} unique indices and expected: {x.size}")
    return


def classify_and_sort(dloader: DataLoader, n_clusters: int) -> np.ndarray:
    n_classes = 10

    # Sort the samples by GT identity. Record the index of each sample in each set.
    gt_classes = np.zeros((n_classes, int(len(dloader)/n_classes)), dtype=np.int64)
    class_ptr = np.zeros((n_classes,), dtype=np.int64)

    for i, s in enumerate(dloader):
        # each sample is [sample, gt]
        classid = s[1].cpu().numpy()[0]
        ptr = class_ptr[classid]
        gt_classes[classid][ptr] = i
        class_ptr[classid] += 1
    
    print(f"Got {gt_classes.shape[1]} samples for {n_classes} classes")
    # SANITY: check for uniqueness
    check_uniqueness(gt_classes)

    kmeans = KMeans(n_clusters=n_clusters).half().to(torch_device)
    c_kmeans = torch.compile(kmeans)
    if False:
        # show a summary
        summary(kmeans, (4, 250, 512))

    chosen_samples = np.zeros((n_classes, n_clusters), dtype=np.int64)

    for category in range(n_classes):
        class_samples = get_samples(dloader, sample_list=gt_classes[category])

        # Compute embeddings for each sample.
        emb = clip_embeddings(class_samples).half()
        # Be conservative with memory
        del class_samples
        
        print(f"Computing top-{n_clusters} feature clusters for class: {category} ...")
        # Run k-means on all embeddings
        category_kmeans = deepcopy(c_kmeans).to(torch_device)
        cluster_idx = category_kmeans.fit_predict(emb)
        cluster_idx = torch.flatten(cluster_idx).cpu().numpy()
        # This has shape (1, n_samples)
        # 
        # Here we will take the simplest approach of choosing the first element for each k-bin.
        # The more optimal version would select the sample with the lowest distance from 
        # each k-centroid.
        #
        # Once we get indices into the categorical subset, we need to remap those indices into
        # the full set.
        category_indices = np.array([int(np.where(cluster_idx == i)[0][0]) for i in range(n_clusters)])
        # check_uniqueness(category_indices)
        chosen_samples[category] = gt_classes[category][category_indices]
        # print("")

    check_uniqueness(chosen_samples)
    return chosen_samples


def write_samples(chosen_samples: np.ndarray):
    # Write out our chosen samples. We will do this using a very simple scheme that's easy for
    # a dataset wrapper to consume. We will map from a set of included samples onto an index
    # from the original dataset.        
    fname = f"mini_samples.json"
    print(f"Writing mini samples to file: {fname}")
    with open(fname, 'w') as f:
        json.dump(chosen_samples, f, indent=4)
    return


def main(args: Namespace):
    train_dataloader, test_dataloader = build_dataloaders(args)

    subset_map = {
        'train': train_dataloader,
        'test': test_dataloader,
    }

    mini_samples = {
        'train': [],
        'test': [],
    }

    for k, v in subset_map.items():
        n_samples = int(args.sample_pct * len(v))
        print(f"Distilling subset: {k} to {n_samples} samples per class")
        chosen_samples = classify_and_sort(v, n_clusters=n_samples)

        # Put the results in a simple data struct
        for category in range(len(chosen_samples)):
            for s in chosen_samples[category]:
                mini_samples[k].append(int(s))
        check_uniqueness(mini_samples[k])

    write_samples(mini_samples)
    return


def get_args():
    parser = argparse.ArgumentParser(description=f"Minify CIFAR10")
    parser.add_argument('-b', '--batch', type=int, default=1, help='Batch size')
    parser.add_argument('-d', '--data', type=str, default="./data", help='Dataset location')
    parser.add_argument('-s', '--sample-pct', type=float, default=0.002, help='Percentage of samples to keep')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use')
    all_args = parser.parse_args()
    return all_args


if __name__ == "__main__":
    run_args = get_args()
    torch_device = torch.device(run_args.device)
    print(f"Using pytorch device: {torch_device}")
    main(run_args)
