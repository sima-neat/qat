# `sima_qat.sima_quantizer`

Source: `sima_qat/sima_quantizer.py`

## Public API

### Class: `SimaMovingAverageMinMaxObserver`

We override the vanilla Pytorch MinMaxObserver with a specialized version. This
specialized version averages min/max over each sample in the batch. This gives
more consistency across samples, and makes training behavior more independent
from the batch size setting.

### Function: `get_sima_quantization_config(is_qat, shift_aware)`

No docstring available.

### Class: `SimaQuantizer`

This quantizer definition uses XNNPACK implementation for the majority of ops. This is because
the XNNPACK code looks for appropriate patterns and applies the QuantizationAnnotation
attributes accordingly. The quantization rules are defined separately, and specified here using
Sima properties.
