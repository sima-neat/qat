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
    Dataset loading and iteration uses skimage.io; the rest of 
    the inference path uses only numpy / sklearn / onnxruntime.
"""
import sys
import os
import datetime
from glob import glob
from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, List, Iterable
from pathlib import Path
import logging
import time

from tqdm import tqdm
import onnx
import onnxruntime
import numpy as np
import cv2

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as L

from sima_qat.misc import find_latest_file_string


def validate_model_and_input(onnx_file_name: str, input_: np.ndarray) -> np.ndarray:
    # Find out if we are using a more recent ONNX model.
    onnx_model = onnx.load(onnx_file_name)
    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    # get the name of the first input of the model
    input_t = ort_session.get_inputs()[0]
    logging.info(f"Input {input_t.name}, shape: {input_t.shape}")
    output_t = ort_session.get_outputs()[0]
    logging.info(f"Output {output_t.name}, shape: {output_t.shape}")

    ort_inputs = {input_t.name: input_}
    outs = ort_session.run(None, ort_inputs)
    return outs


class MNISTIterator(object):
    """ This is a helper class to iterate over samples in the MNIST dataset.

        Note: __iter__ not implemented for now.
    """
    def __init__(self, ds_root: str, download: bool) -> None:

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        assert os.path.isdir(ds_root)

        # Fixed to test set for now.
        self.mnist_dataset_test = MNIST(
            download=download, 
            root=ds_root, 
            transform=transform,
            train=False,
        )
        return

    def __len__(self) -> int:
        return len(self.mnist_dataset_test)

    def __getitem__(self, i: int) -> Dict:
        """ Returns dict: {'image', 'mask'}
        """
        v = self.mnist_dataset_test[i]
        return {'sample': v[0], 'gt': v[1]}
    

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


def debug_compare_tensors(t_dut: np.ndarray, t_ref_name: str) -> bool:
    """ Do a comparison against a reference value.
    """
    if os.path.isfile(t_ref_name):
        hmap_ref = np.load(t_ref_name)
    else:
        logging.warning(f"No numpy debug file found for: {t_ref_name}")
    # We will print a message at > 10%. There seems to be a surprising difference
    # for some elements in each tensor, so this should prompt some additional debug
    # to find out more.
    return debug_allclose(hmap_ref, t_dut, rtol=1e-1)


def run_accuracy_test(
        ort_session: onnxruntime.InferenceSession,
        dataset_test: object,
    ) -> float:
    """ Run the dataset against the model and compute accuracy.
    """
    super_debug = False

    # Get model IO
    input_t = ort_session.get_inputs()[0]
    output_t = ort_session.get_outputs()[0]

    l = len(dataset_test)
    class_outputs = np.zeros((l,), dtype=np.int32)
    class_gt = np.zeros((l,), dtype=np.int32)
    inf_start = time.perf_counter()
    # For some strange reason, the builtin iterator throws an exception because
    # it tries to iterate past the dataset size ...
    for i in tqdm(range(l)):
        sample = dataset_test[i]
        # Expand the batch dimension to 1: (1, H, W) -> (1, 1, H, W)
        nn_in = np.expand_dims(sample['sample'], axis=0)
        s_out = ort_session.run(None, {input_t.name: nn_in})
        # 1st network output, batch dim = 1
        net_map = s_out[0][0]
        class_outputs[i] = np.argmax(net_map)
        class_gt[i] = sample['gt']

    inf_end = time.perf_counter()
    fps = l / (inf_end - inf_start)
    logging.info(f"FP32 FPS: {fps}")

    scores = class_outputs == class_gt
    acc = np.mean(scores.astype(np.float32))
    return acc



def get_args() -> Namespace:
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    # Find the presence of onnx files first
    recent_onnx_file = find_latest_file_string(os.getcwd())

    parser = ArgumentParser()
    parser.add_argument("--onnx", type=str, required=False, default=recent_onnx_file, help="The ONNX file containing a MNIST Model.")
    parser.add_argument("--dsroot", type=str, required=False, default='.', help="Directory for the root of the dataset.")
    parser.add_argument('--download', action='store_true', help='Download dataset to specified data path')
    parser.add_argument('-v', '--verbosity', type=str, default='INFO', help='Logging verbosity level')
    return parser.parse_args()


def main():
    args = get_args()
    logging.getLogger().setLevel(args.verbosity)

    # Set the global seed to be able to be able to replicate results.
    L.seed_everything(42)

    onnxf = os.path.abspath(args.onnx)
    logging.info(f"Loading ONNX file: {onnxf}")

    dim = 28
    input_ = np.random.rand(1, 1, dim, dim).astype(np.float32)
    logging.info(f'Testing model on input shape: {input_.shape}')

    outs = validate_model_and_input(onnxf, input_)
    logging.info('Succeeded.')

    # After we do a smoke test of the model (does it compile and produce outputs), we pass
    # dataset samples into the network and manually compute accuracy.
    dsroot = os.path.abspath(args.dsroot)
    logging.info(f"Using MNIST dataset at: {dsroot}")
    dataset_test = MNISTIterator(ds_root=dsroot, download=args.download)

    ort_session = onnxruntime.InferenceSession(onnxf, sess_opts=None)
    acc = run_accuracy_test(
        ort_session, 
        dataset_test, 
    )
    logging.info(f"Top-1 accuracy: {acc}")


if __name__ == '__main__':
    main()