---
title: Operator annotation contract
sidebar_position: 3
---

# Operator annotation contract

The versioned operator manifest records the PyTorch module and functional
forms, captured ATen forms, QAT behavior, expected opset-17 ONNX operators, and
test cases for each operator family. It describes QAT training and standard
QDQ export behavior. It does not claim that a downstream compiler assigns an
operator to a particular hardware engine.

## Status

Each manifest entry has one of these statuses:

| Status | Meaning |
| --- | --- |
| `supported` | The declared captured form has lifecycle and export coverage. |
| `partial` | Only the listed topology or operand forms are covered. Other forms remain outside the contract. |
| `deferred` | QAT does not claim the form yet. It is either left trainable and unannotated or awaits composite coverage. |

## Behavior

Status and behavior answer different questions. A supported operator can
participate in QAT in several ways:

| Behavior | Meaning |
| --- | --- |
| `annotation` | The operator receives explicit fake-quant input, output, or weight annotations. |
| `propagation` | A shape or layout operation preserves an existing activation grid. |
| `type-preserving` | A tensor stays quantized while a non-floating output, such as an index, keeps its original dtype. |
| `mixed-output` | Only eligible floating outputs participate in QAT; index or metadata outputs remain unquantized. |
| `decomposed` | PyTorch or ONNX represents the operation as a tested sequence of simpler operators. |
| `training-only` | The operation participates in training and is folded or finalized for inference. |
| `passthrough` | The operation remains trainable but receives no QAT annotation. |

For example, `ArgMax` and `TopK` keep their PyTorch and ONNX index outputs as
`int64`; fake quantization is never attached to those index values. Exact GELU
and SiLU have explicit decomposed contracts. Flatten, Transpose, Pad, and other
grid-preserving operations propagate an existing activation grid without
inventing a new one.

PReLU, ConvTranspose, Embedding/Gather, GridSample, ReduceMin, and CumSum are
intentional pass-through forms. Their presence does not reject the whole model,
but QAT does not claim fake-quantized behavior for those operations.

## Inspect the installed manifest

The installed package contains the machine-readable manifest and its version:

```python
from sima_qat.operator_manifest import (
    ONNX_OPSET,
    OPERATOR_MANIFEST,
    OPERATOR_MANIFEST_VERSION,
)

print(OPERATOR_MANIFEST_VERSION, ONNX_OPSET)
for entry in OPERATOR_MANIFEST:
    print(entry.family, entry.status.value, entry.behavior.value)
```

Consult each entry's `operand_constraints`, `captured_aten_forms`, and test case
IDs before treating a broader PyTorch spelling or topology as covered.
