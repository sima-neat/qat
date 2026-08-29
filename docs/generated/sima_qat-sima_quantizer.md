# `sima_qat.sima_quantizer`

Source: `sima_qat/sima_quantizer.py`

## Public API

### Class: `SimaMovingAverageMinMaxObserver`

We override the vanilla Pytorch MinMaxObserver with a specialized version. This
specialized version averages min/max over each sample in the batch. This gives
more consistency across samples, and makes training behavior more independent
from the batch size setting.

### Class: `FullRangeSTEFakeQuantize`

Fake INT8 forward with an identity backward, including clipped values.

PyTorch's native fake-quant backward zeroes gradients outside the observed
range. DepthART's recurrent state can temporarily exceed that range early
in QAT, which otherwise blocks the very gradient needed to pull it back.
The forward remains bit-identical to native fake quantization; only the
training surrogate gradient changes.

### Function: `get_sima_quantization_config(is_qat, shift_aware, activation_observer, full_range_ste, learn_scales)`

No docstring available.

### Class: `SimaQuantizer`

This quantizer definition uses XNNPACK implementation for the majority of ops. This is because
the XNNPACK code simply looks for appropriate patterns, and applies the QuantizationAnnotation
attributes accordingly. The quantization rules are defined separately, and specified here using
Sima properties.
