#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
""" Script to load a generated ONNX file and run test samples through it.
    The script first searches the most recent ONNX model and then loads the validation data. 
    Executes the ONNX file in onnxruntime / numpy. 
"""
import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import onnx
import logging
import time
import onnxruntime
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser, Namespace

from typing import Callable, Dict, List, Iterable

from sima_qat.misc import find_latest_file_string

from imagenet_dataset import (
    apply_imagenet_target_transform,
    limit_samples_by_class,
    set_dataset_samples,
)

import pytorch_lightning as L


# Helper class to iterate over ImageNet samples
class ImageNetIterator(object):
    """Helper class to iterate over ImageNet-style split folders."""

    def __init__(self, ds_root: str, split: str = 'val', samples_limit: int | None = None) -> None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        split_dir = os.path.join(ds_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"ImageNet split directory not found: {split_dir}. "
                f"Expected a dataset layout like {ds_root}/train and {ds_root}/val."
            )

        self.imagenet_dataset = datasets.ImageFolder(split_dir, transform=transform)
        apply_imagenet_target_transform(self.imagenet_dataset)
        if samples_limit is not None:
            set_dataset_samples(
                self.imagenet_dataset,
                limit_samples_by_class(self.imagenet_dataset.samples, samples_limit),
            )

    def __len__(self) -> int:
        return len(self.imagenet_dataset)

    def __getitem__(self, i: int) -> Dict:
        """Returns dict: {'sample': image, 'gt': label}."""
        v = self.imagenet_dataset[i]
        return {'sample': v[0], 'gt': v[1]}


# Validate model and input with ONNX runtime
def validate_model_and_input(onnx_file_name: str, input_: np.ndarray) -> np.ndarray:
    onnx_model = onnx.load(onnx_file_name)
    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    input_t = ort_session.get_inputs()[0]
    logging.info(f"Input {input_t.name}, shape: {input_t.shape}")
    output_t = ort_session.get_outputs()[0]
    logging.info(f"Output {output_t.name}, shape: {output_t.shape}")

    ort_inputs = {input_t.name: input_}
    outs = ort_session.run(None, ort_inputs)
    return outs


def debug_allclose(a, b, rtol=1e-2) -> bool:
    ''' Helper function.
    '''
    a = a.flatten()
    b = b.flatten()
    err = False
    for i in range(len(a)):
        if b[i] == 0.0:
            diff = a[i] - b[i]
        else:
            diff = abs((a[i]/(b[i]+1e-9)) - 1.0)
        if diff > rtol:
            print(f"Got [{i}] miscompare: {diff:.6f} pct, {a[i]} / {b[i]}")
            # return False
            err = True
    return not err

# Debug function to compare tensors
def debug_compare_tensors(t_dut: np.ndarray, t_ref_name: str) -> bool:
    """Compare model output with reference value."""
    if os.path.isfile(t_ref_name):
        hmap_ref = np.load(t_ref_name)
    else:
        logging.warning(f"No numpy debug file found for: {t_ref_name}")
    return debug_allclose(hmap_ref, t_dut, rtol=1e-1)


# Accuracy test function
def run_accuracy_test(ort_session: onnxruntime.InferenceSession, dataset_test: object) -> float:
    """Run the dataset against the model and compute accuracy."""
    input_t = ort_session.get_inputs()[0]
    output_t = ort_session.get_outputs()[0]

    l = len(dataset_test)
    class_outputs = np.zeros((l,), dtype=np.int32)
    class_gt = np.zeros((l,), dtype=np.int32)
    inf_start = time.perf_counter()

    for i in tqdm(range(l)):
        sample = dataset_test[i]
        nn_in = np.expand_dims(sample['sample'], axis=0)  # Add batch dimension
        s_out = ort_session.run(None, {input_t.name: nn_in})
        net_map = s_out[0][0]
        class_outputs[i] = np.argmax(net_map)
        class_gt[i] = sample['gt']

    inf_end = time.perf_counter()
    fps = l / (inf_end - inf_start)
    logging.info(f"FP32 FPS: {fps}")

    scores = class_outputs == class_gt
    acc = np.mean(scores.astype(np.float32))
    return acc


# CLI argument parser
def get_args() -> Namespace:
    """Get CLI arguments."""
    recent_onnx_file = find_latest_file_string(os.getcwd())

    parser = ArgumentParser()
    parser.add_argument("--onnx", type=str, required=False, default=recent_onnx_file, help="The ONNX file containing the ImageNet Model.")
    parser.add_argument("--dsroot", type=str, required=False, default='.', help="Directory for the root of the dataset.")
    parser.add_argument('--split', type=str, default='val', help="Dataset split (test or val).")
    parser.add_argument('--samples-limit', type=int, default=None, help='Limit evaluation samples to size N')
    parser.add_argument('-v', '--verbosity', type=str, default='INFO', help='Logging verbosity level')
    return parser.parse_args()


# Main function
def main():
    args = get_args()
    logging.getLogger().setLevel(args.verbosity)

    if not args.onnx:
        raise FileNotFoundError("No ONNX file found. Pass --onnx or run training/export first.")

    # Set the global seed to replicate results.
    L.seed_everything(42)

    onnxf = os.path.abspath(args.onnx)
    logging.info(f"Loading ONNX file: {onnxf}")

    input_ = np.random.rand(1, 3, 224, 224).astype(np.float32)  # ImageNet input size
    logging.info(f'Testing model on input shape: {input_.shape}')

    outs = validate_model_and_input(onnxf, input_)
    logging.info('Succeeded.')

    # Load dataset and run accuracy test
    dsroot = os.path.abspath(args.dsroot)
    logging.info(f"Using ImageNet dataset at: {dsroot}")
    dataset_test = ImageNetIterator(ds_root=dsroot, split=args.split, samples_limit=args.samples_limit)

    ort_session = onnxruntime.InferenceSession(onnxf, sess_opts=None)
    acc = run_accuracy_test(ort_session, dataset_test)
    logging.info(f"Top-1 accuracy: {acc}")


if __name__ == '__main__':
    main()