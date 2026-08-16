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

import pytorch_lightning as L
from pytorch_lightning.utilities import disable_possible_user_warnings

if __package__:
    from .cifar_classifier import get_args, training_test
else:
    from cifar_classifier import get_args, training_test


# Absolute path to the cached CIFAR dataset (next to this test), so it is found regardless
# of the test's working directory (see tests/end_to_end/conftest.py); downloaded on demand.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.regression
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

