import os
import argparse
from argparse import Namespace
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pytest

import torch
from torchvision.models import DenseNet
from torch import optim, nn, utils, Tensor

pytest.importorskip("pytorch_lightning")
import pytorch_lightning as L
from pytorch_lightning.utilities import disable_possible_user_warnings

from cifar_classifier import get_args, training_test


# Absolute path to the cached CIFAR dataset (next to this test), so it is found regardless
# of the test's working directory (see tests/conftest.py); downloaded on demand.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@pytest.mark.nightly
def test_densenet():
    disable_possible_user_warnings()
    torch.set_num_threads(1)    # Important to tame resources during regressions
    args = Namespace(
        epochs=4,
        batch=16,
        data=_DATA_DIR,
        device='cpu',
        disable_qat=False,
        lr=5e-3,
        acc=0.15,
    )
    rc = run_densenet(args)
    assert rc == True
    return


def run_densenet(args: Namespace):
    # This is a super minified version of the original DenseNet. It's intended to test a small 
    # sub-section of the DenseNet architecture for QAT amenity.
    densenet_model = DenseNet(
        growth_rate=4, 
        block_config=(6, 0, 0, 0), 
        num_init_features=16, 
        bn_size=2, 
        num_classes=10
    )
    rc = training_test(args=args, classifier_model=densenet_model, model_name='DenseNetFragment')
    return rc


if __name__ == "__main__":
    run_args = get_args()

    # The command line invocation uses the generic args version.
    run_densenet(run_args)
