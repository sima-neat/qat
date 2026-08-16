# `sima_qat.sima_quantizer`

Source: `sima_qat/sima_quantizer.py`

SiMa signed-int8 configuration for PyTorch FX graph-mode QAT.

This module deliberately contains no PT2E quantizer annotations and no Dynamo
imports. The model-compiler environment uses Python 3.12 with PyTorch 2.3,
where Dynamo graph capture is unavailable. FX graph-mode QAT uses the symbolic
FX tracer instead and is supported by that stack.

## Public API

### Constant: `ACTIVATION_QUANT_MAX = 127`

### Constant: `ACTIVATION_QUANT_MIN = -128`

### Constant: `QPARAM_EPS = 2 ** (-12)`

### Class: `SimaMovingAverageMinMaxObserver(MovingAverageMinMaxObserver)`

Activation observer which averages sample extrema within each batch.

Averaging per-sample extrema makes calibration less sensitive to the batch
size while retaining the moving-average behavior used by QAT.

### Constant: `WEIGHT_QUANT_MAX = 127`

### Constant: `WEIGHT_QUANT_MIN = -127`

### Function: `get_sima_backend_config() -> BackendConfig`

Return the FX fusion and operator-pattern registry used by SiMa QAT.

QNNPACK supplies the portable FX pattern table. SiMa overrides observer
sharing at concat, unsqueeze, indexing, and SiLU boundaries to retain the
quantization regions produced by the previous backend.

### Function: `get_sima_qconfig() -> QConfig`

Return the signed-int8 QAT configuration used by SiMa models.

### Function: `get_sima_qconfig_mapping() -> QConfigMapping`

Return the global FX QAT mapping for supported operator patterns.
