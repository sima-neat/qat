# QAT production-readiness handoff

## Purpose and branch

Continue from branch `feat/afe-supported-qat-operators`.

This branch keeps the QAT repository responsible for PyTorch capture,
quantization annotations, fake quantization, freezing, finalization, and
opset-17 QDQ export. Graph surgery, AFE tests in QAT CI, MLA execution, and
postprocessing remain outside this repository.

The compiler reference used by the operator manifest is
[`sima-neat/model-compiler` PR 111](https://github.com/sima-neat/model-compiler/pull/111),
revision `7b9cf790aed8367f93ebd3a74064e200ac2fdc08`, schema 4, release 2.1.
The AFE master revision inspected during this work was
`89c3020e29b50ad656ba9a2fee40c2fac6cce78b`.

## What is implemented

- `sima_qat/operator_manifest.py` is the versioned source of truth for the
  opset-17 QAT contract.
- The manifest contains 67 entries and preserves the exact 59-operator INT8,
  opset-17 compiler snapshot from PR 111.
- Every entry records PyTorch modules, functional APIs, captured ATen forms,
  annotation behavior, dtypes, constraints, annotators, expected ONNX
  operators, test requirements, and implementation status.
- The operator matrix has grown from 33 to 72 executable PyTorch cases.
- Added real PT2E annotations for unary math/activations, binary arithmetic,
  Einsum, PReLU, Pow, reductions, global max pooling, resize, ArgMax/TopK, and
  grid-preserving layout operations.
- Structural operations share an existing activation grid instead of creating
  gratuitous requantization boundaries.
- ArgMax and TopK quantize floating inputs/values but do not fake-quantize
  integer indices.
- The following W8A8 forms now fail explicitly instead of silently producing
  an unquantized graph: ConvTranspose2d with the unresolved weight-axis
  contract, Embedding/Gather, GridSample, ReduceMin, and CumSum.
- ONNX parity cases are derived from the manifest rather than a manually
  maintained family subset.

## Local test evidence

Use the repository's Python 3.12/Torch 2.8 environment:

```bash
cd qat
PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD:$PWD/../afe_env/lib/python3.12/site-packages" \
../../../python312/bin/python3.12 -m pytest <paths> -q
```

Results collected on this branch:

- Complete regression suite: `320 passed, 3 skipped` in 10 minutes 20 seconds.
- Annotation, freeze, and finalization matrix: `179 passed`.
- Manifest drift guards: `6 passed`.
- Operator contracts plus the original representative ONNX suite: `31 passed`.
- Expanded opset-17 ONNX checker/Runtime matrix: `72 passed`. The test allows
  two output quanta because independent QDQ rounding can differ at more than
  one edge; the tolerance remains tied to each exported quantization grid.

Known non-fatal warnings remain:

- Torch 2.8 PT2E deprecation warnings.
- Duplicate BatchNorm deletion warnings for Conv-BN patterns.
- InstanceNorm export warns that its captured operator remains in training
  mode; verify or correct inference-mode export before release.
- Legacy TorchScript ONNX exporter deprecation warnings.

## Required Model SDK 2.1.3 / AFE master verification

Run this manually in the known-good compiler environment. Do not add this
compiler dependency to QAT CI.

### 1. Regenerate the persistent QDQ corpus

```bash
cd qat
rm -rf /tmp/sima-qat-opset17-corpus
PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD:$PWD/../afe_env/lib/python3.12/site-packages" \
../../../python312/bin/python3.12 -m pytest \
  tests/operators/test_onnx_export.py -q \
  --basetemp=/tmp/sima-qat-opset17-corpus
find /tmp/sima-qat-opset17-corpus -name '*.onnx' -print | sort
```

Expected result: all 72 cases pass ONNX checker and ONNX Runtime parity and 72
ONNX files are present.

### 2. Verify the exact compiler environment

Record all of the following in the validation report:

- Model SDK/compiler release: 2.1.3.
- AFE commit: master at or after `89c3020e`.
- TVM, MLA compiler, and kernel package revisions selected by that SDK.
- Host architecture and Python version.
- The QAT branch commit under test.

### 3. Import and partition every exported model

For every ONNX file:

1. Confirm the default-domain opset is 17.
2. Run ONNX validation.
3. Import through Model SDK/AFE using the shapes and dtypes stored in the ONNX
   interface.
4. Run AFE transformation and partitioning.
5. Record every ONNX operator, the resulting AFE operators, backend assignment,
   inserted casts/requantization, and any APU fallback.
6. Treat an importer error, unsupported operator, incorrect integer-index
   handling, or unexpected APU fallback as a manifest failure.

Pay particular attention to:

- `ArgMax` and `TopK`: QAT preserves PyTorch `int64` indices, while MLA emits
  `int32`. Downstream consumers must have a documented cast contract.
- `PRelu`: its 1D alpha uses a per-tensor activation-style qspec because the
  Conv/Linear per-channel observer cannot operate on a rank-1 tensor. Confirm
  compiler ingestion and numerical behavior.
- `Flatten`: PR 111 records schema 21, while this repository exports the
  opset-17 schema. Verify the older schema explicitly.
- `Pad`: only constant-zero padding across at most two dimensions is intended.
- `Resize`: verify both nearest and linear modes against the compiler scaling
  constraints.
- `Softmax`, reductions, Transpose, and TopK: verify automatic layout
  conversion maps logical axes to supported MLA axes without moving batch.
- Exact GELU: verify the opset-17 `Div/Erf/Add/Mul` decomposition, not native
  opset-20 `Gelu`.
- SiLU: verify the `Sigmoid/Mul` decomposition.
- BatchNorm: it must be frozen/folded for deployment and not treated as a
  standalone INT8 compiler kernel.

### 4. Compile representative models

Import coverage alone is insufficient. Compile at least one representative
from each behavior class:

- Weighted annotation: Conv2d and Linear.
- Multi-input arithmetic: MatMul, Add, Div, and Einsum.
- Lookup/nonlinear: Erf, Exp, Log, Sigmoid, and Tanh.
- Reduction: ReduceMean, ReduceSum, ReduceMax, and ReduceLogSumExp.
- Grid propagation: Reshape, Transpose, Split, Pad, DepthToSpace, and Tile.
- Mixed outputs: ArgMax and TopK.
- Normalization: LayerNormalization, InstanceNormalization, and PRelu.

Where simulator execution is available, compare compiled outputs against ONNX
Runtime using quantization-aware tolerances and retain the artifacts/report.

## Manifest entries that are not production-complete

### Deferred

- `ConvTranspose`: implement the correct output-channel per-channel weight
  qspec, including grouped and depthwise cases, then extend shift-aware freeze.
- `LRN`: Torch capture decomposes it through `avg_pool3d`; the complete
  composite is not annotated or tested.
- `MeanVarianceNormalization`: no stable single PyTorch/ATen form is covered.
- `ReduceLogSum` and `ReduceSumSquare`: add composite lifecycle and ONNX tests.
- Variadic ONNX `Mean` and `Sum`: add explicit multi-input PyTorch/export
  topologies.

### Partial

- `Relu`: fused forms are covered; standalone and fan-out topologies remain.
- `Clip`/Hardtanh: Conv/Add fused forms are covered; standalone clamp and
  arbitrary bounds remain.
- `Slice`: the slice-select-unsqueeze composite is covered; general positive
  slices, view-like forms, and negative-step rejection remain.

### Explicitly rejected for W8A8

- Embedding/Gather.
- GridSample.
- ReduceMin.
- CumSum.
- ConvTranspose until its weight-grid contract is corrected.

## Work required after compiler verification

1. Convert every compiler constraint in the manifest into executable positive
   and negative tests. The manifest currently records all constraints, but not
   every rank/axis/broadcast/exponent/mode failure is enforced by validation.
2. Assert expected ONNX operator families in export tests. Current tests check
   valid QDQ and Runtime parity but do not yet compare every exported node set
   with `expected_onnx_operators`.
3. Resolve the deferred and partial entries above. Do not change them to
   `SUPPORTED` based only on successful ONNX export.
4. Fix the duplicate BatchNorm-node deletion warnings and require warning-free,
   idempotent finalization.
5. Run the complete CPU regression suite from a built wheel and add formatting,
   lint, type-check, package-import, and documentation-link gates.
6. Add the scheduled CUDA QAT smoke test. It must cover device movement,
   backward, freeze, finalization, and export without invoking AFE.
7. Finish reproducible packaging, dependency separation, semantic versioning,
   release notes, and tag-to-package promotion.
8. Add the `sima-qat` skill and integrate canonical QAT documentation into
   `core`; keep graph surgery and deployment postprocessing downstream.

The broader checklist remains in `PRODUCTION_READINESS_TODO.md`.
