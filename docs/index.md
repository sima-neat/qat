---
title: Quantization-Aware Training
sidebar_label: Overview
sidebar_position: 1
---

# Quantization-Aware Training

SiMa QAT prepares PyTorch models for quantization-aware training and exports
standard opset-17 ONNX models with QuantizeLinear and DequantizeLinear (QDQ)
nodes. Use it when training can recover accuracy after introducing INT8
quantization effects. Use Model Compiler post-training quantization (PTQ) when
you have a completed floating-point ONNX model and representative calibration
data but do not plan to retrain it.

QAT owns the PyTorch training and ONNX export workflow. Importing, optimizing,
or compiling the exported ONNX model is a separate Model Compiler step.

## Requirements

| Component | Supported version |
| --- | --- |
| Python | 3.10 or newer; release CI uses Python 3.12 |
| PyTorch | 2.8.x; the wheel pins 2.8.0 |
| ONNX | 1.17.0 |
| ONNX Runtime | 1.21.1 |

Install the wheel into a PyTorch training environment that also contains the
model's normal training dependencies.

## Install

Download the wheel for a release, tag, or branch with `sima-cli`:

```bash
sima-cli neat install qat@<release-or-branch>
```

The command downloads the QAT wheel without changing the active Python
environment. Change to the download directory, activate the environment used
to train the model, and install the downloaded wheel:

```bash
python -m pip install ./sima_qat-*.whl
```

Verify that the package and required PyTorch version load together:

```bash
python -c "import torch, sima_qat; print(torch.__version__, sima_qat.__file__)"
```

## Continue

- Follow the [training lifecycle](./training-lifecycle.md) to prepare, train,
  freeze, resume, finalize, and export a model.
- Read the [operator annotation contract](./operator-support.md) to understand
  which PyTorch forms receive fake quantization and which remain unannotated.
- After export, continue with
  [Model compilation](/compile-a-model/model-compilation/).
