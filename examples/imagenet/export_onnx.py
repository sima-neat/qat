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

from imagenet_lit import ImageNet_Model_Trainer


def run_export(args: Namespace):
    """ Run the training regimen.
    """
    print(f"Loading checkpoint file: {args.ckpt}")
    classifier = ImageNet_Model_Trainer.load_from_checkpoint(args.ckpt)
    classifier.to(args.device)
    classifier._finalize_qat_model()
    classifier.to_onnx(f"exported_ckpt_{args.model}_onnx_model.onnx")


def find_latest_file_string(root_path: str, model_name: str, tag_str: str='.ckpt') -> str:
    ''' 
    Recursively find the most recent tag_str file in the specified path
    that also matches the model name. 
    '''
    most_recent_dt = None
    most_recent_file = None
    for f_dir, f_subdirs, f_names in os.walk(root_path):
        # Filter files that contain both the model_name and tag_str
        for f in [fx for fx in f_names if tag_str in fx and model_name in fx]:
            visit_file = os.path.join(f_dir, f)
            m_time = os.path.getmtime(visit_file)
            # Convert timestamp into DateTime object
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
    parser = argparse.ArgumentParser(description="Download ONNX file for most recent checkpoint")
    # Add model name as an argument
    parser.add_argument('--model', type=str, required=True, help='Model name to search for in checkpoint files')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use')
    
    # Parse arguments initially to get the model name
    all_args, unknown = parser.parse_known_args()
    
    # Find the latest checkpoint based on model_name
    latest_ckpt = find_latest_file_string('.', model_name=all_args.model)
    
    # Re-parse arguments, this time adding the checkpoint as default
    parser.add_argument('-c', '--ckpt', type=str, default=latest_ckpt, help='Checkpoint to load')
    all_args = parser.parse_args()

    return all_args


if __name__ == "__main__":
    # Set the global seed to prevent headaches.
    L.seed_everything(42)
    run_args = get_args()

    run_export(run_args)

