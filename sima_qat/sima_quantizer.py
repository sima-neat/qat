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
# SiMa.ai. Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure of this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.
#**************************************************************************
"""SiMa signed-int8 configuration for PyTorch FX graph-mode QAT.

This module deliberately contains no PT2E quantizer annotations and no Dynamo
imports. The model-compiler environment uses Python 3.12 with PyTorch 2.3,
where Dynamo graph capture is unavailable. FX graph-mode QAT uses the symbolic
FX tracer instead and is supported by that stack.
"""

from __future__ import annotations

import copy
import operator

import torch
from torch.ao.quantization import FakeQuantize, QConfig, QConfigMapping

# PyTorch 2.3's qnnpack factory expects this submodule to have been imported
# before it dereferences torch.ao.quantization.backend_config.utils.
import torch.ao.quantization.backend_config.utils as _backend_config_utils  # noqa: F401
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    ObservationType,
    get_qnnpack_backend_config,
)
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)


ACTIVATION_QUANT_MIN = -128
ACTIVATION_QUANT_MAX = 127
WEIGHT_QUANT_MIN = -127
WEIGHT_QUANT_MAX = 127
QPARAM_EPS = 2**-12


__all__ = [
    "ACTIVATION_QUANT_MAX",
    "ACTIVATION_QUANT_MIN",
    "QPARAM_EPS",
    "SimaMovingAverageMinMaxObserver",
    "WEIGHT_QUANT_MAX",
    "WEIGHT_QUANT_MIN",
    "get_sima_backend_config",
    "get_sima_qconfig",
    "get_sima_qconfig_mapping",
]


class SimaMovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
    """Activation observer which averages sample extrema within each batch.

    Averaging per-sample extrema makes calibration less sensitive to the batch
    size while retaining the moving-average behavior used by QAT.
    """

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)
        if self.min_val == float("inf") and self.max_val == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            if x.ndim < 2:
                min_val_cur, max_val_cur = torch.aminmax(x)
            else:
                x_flat = x.reshape(x.shape[0], -1)
                min_val_batch, max_val_batch = torch.aminmax(x_flat, dim=1)
                min_val_cur = min_val_batch.mean()
                max_val_cur = max_val_batch.mean()

            min_val = self.min_val + self.averaging_constant * (
                min_val_cur - self.min_val
            )
            max_val = self.max_val + self.averaging_constant * (
                max_val_cur - self.max_val
            )

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


def get_sima_qconfig() -> QConfig:
    """Return the signed-int8 QAT configuration used by SiMa models."""

    activation = FakeQuantize.with_args(
        observer=SimaMovingAverageMinMaxObserver,
        quant_min=ACTIVATION_QUANT_MIN,
        quant_max=ACTIVATION_QUANT_MAX,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        eps=QPARAM_EPS,
    )
    weight = PerChannelMinMaxObserver.with_args(
        quant_min=WEIGHT_QUANT_MIN,
        quant_max=WEIGHT_QUANT_MAX,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        reduce_range=False,
        eps=QPARAM_EPS,
    )
    return QConfig(activation=activation, weight=weight)


def get_sima_qconfig_mapping() -> QConfigMapping:
    """Return the global FX QAT mapping for supported operator patterns."""

    return QConfigMapping().set_global(get_sima_qconfig())


def get_sima_backend_config() -> BackendConfig:
    """Return the FX fusion and operator-pattern registry used by SiMa QAT.

    QNNPACK supplies the portable FX pattern table. SiMa overrides observer
    sharing at concat, unsqueeze, indexing, and SiLU boundaries to retain the
    quantization regions produced by the previous backend.
    """

    backend_config = get_qnnpack_backend_config()
    configs = list(backend_config.configs)

    for config in configs:
        if config.pattern is torch.cat:
            config.set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
        elif config.pattern is torch.unsqueeze:
            config.set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )

    unsqueeze_config = next(
        config
        for config in configs
        if config.pattern is torch.unsqueeze
    )
    for pattern in (torch.nn.SiLU, torch.nn.functional.silu):
        silu_config = (
            BackendPatternConfig(pattern)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(copy.deepcopy(unsqueeze_config.dtype_configs))
        )
        backend_config.set_backend_pattern_config(silu_config)

    getitem_config = (
        BackendPatternConfig(operator.getitem)
        .set_observation_type(
            ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
        )
        .set_dtype_configs(copy.deepcopy(unsqueeze_config.dtype_configs))
    )
    backend_config.set_backend_pattern_config(getitem_config)
    return backend_config
