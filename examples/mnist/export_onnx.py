import os
import logging
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import datetime

import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import pytorch_lightning as L

from mnist_lit import MNIST_Trainer


def run_export(args: Namespace):
    """ Run the training regimen.
    """
    print(f"Loading checkpoint file: {args.ckpt}")
    # classifier = MNIST_Trainer()
    classifier = MNIST_Trainer.load_from_checkpoint(args.ckpt)
    classifier.to(args.device)
    classifier._finalize_qat_model()
    classifier.to_onnx("exported_ckpt_onnx_model.onnx")


def find_latest_file_string(root_path: str, tag_str: str='.ckpt') -> str:
    ''' Recursively find the most recent tag_str file in the specified path.
        This function is used to find the most recent output files of a subcommand
        in order to post-process the outputs.
    '''
    most_recent_dt = None
    most_recent_file = None
    for f_dir, f_subdirs, f_names in os.walk(root_path):
        for f in [fx for fx in f_names if tag_str in fx]:
            visit_file = os.path.join(f_dir, f)
            m_time = os.path.getmtime(visit_file)
            # convert timestamp into DateTime object
            dt_m = datetime.datetime.fromtimestamp(m_time)
            if not most_recent_dt:
                most_recent_dt = dt_m
                most_recent_file = visit_file
            else:
                if dt_m > most_recent_dt:
                    most_recent_dt = dt_m
                    most_recent_file = visit_file
    
    return most_recent_file



def get_args():
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    latest_ckpt = find_latest_file_string('.')
    parser = argparse.ArgumentParser(description=f"Download ONNX file for most recent checkpoint")
    parser.add_argument('-c', '--ckpt', type=str, default=latest_ckpt, help='Checkpoint to load')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use')
    all_args = parser.parse_args()
    return all_args


if __name__ == "__main__":
    # Set the global seed to be able to replicate results.
    L.seed_everything(42)
    run_args = get_args()

    run_export(run_args)

