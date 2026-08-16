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
import warnings
from argparse import Namespace
from pathlib import Path

import pytest
import pytorch_lightning as L
import torch
from pytorch_lightning.utilities import disable_possible_user_warnings
from torchvision.models import DenseNet

from .cifar_classifier import training_test


@pytest.mark.slow
@pytest.mark.network
def test_densenet(
    tmp_path: Path,
    cifar_data_dir: Path,
    allow_data_download: bool,
) -> None:
    previous_thread_count = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with warnings.catch_warnings():
            disable_possible_user_warnings()
            args = Namespace(
                epochs=4,
                batch=16,
                data=cifar_data_dir,
                allow_data_download=allow_data_download,
                output_dir=tmp_path,
                device="cpu",
                disable_qat=False,
                lr=5e-3,
                acc=0.15,
                progress_bar=False,
            )
            assert run_densenet(args)
    finally:
        torch.set_num_threads(previous_thread_count)


def run_densenet(args: Namespace) -> bool:
    """Run QAT on a compact DenseNet fragment while preserving DenseNet topology."""
    L.seed_everything(42, workers=True)
    densenet_model = DenseNet(
        growth_rate=4,
        block_config=(6, 0, 0, 0),
        num_init_features=16,
        bn_size=2,
        num_classes=10,
    )
    return training_test(
        args=args,
        classifier_model=densenet_model,
        model_name="DenseNetFragment",
    )
