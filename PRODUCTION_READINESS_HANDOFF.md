# QAT production-readiness handoff

Continue from branch `feat/afe-supported-qat-operators`.

## Scope

This repository owns the PyTorch QAT workflow:

- capture and annotation;
- fake-quantized training and observer updates;
- grid freezing and recovery training;
- finalization;
- standards-compliant opset-17 ONNX QDQ export.

The operator manifest describes this behavior. An entry marked `supported` or
`partial` has executable QAT coverage. An entry marked `deferred` with
`passthrough` behavior remains valid PyTorch, receives no QAT annotation, and
does not gain fake-quant boundaries.

## Current state

- The manifest is versioned and maps every active annotator to its PyTorch,
  captured ATen, and expected ONNX forms.
- The operator matrix exercises prepare, forward/backward, observer freeze,
  finalization, and representative ONNX Runtime parity.
- Exact GELU and SiLU use explicit decompositions for opset 17.
- `ArgMax` and `TopK` preserve their `torch.int64` index outputs without fake
  quantization.
- PReLU, ConvTranspose, Embedding, GridSample, ReduceMin, CumSum, and
  tanh-approximate GELU prepare and train normally but remain unannotated.
- The operator matrix contains 71 opset-17 ONNX checker/runtime cases.
- BatchNorm finalization is warning-free and repeated finalization is
  idempotent.
- InstanceNorm input-statistics behavior is verified against ONNX Runtime
  without the misleading legacy-exporter warning.
- Observer-updating and frozen checkpoints round-trip exactly; frozen qparams
  are preserved through finalization and ONNX export.
- The public-runner GitHub workflow builds an isolated wheel, tests the
  installed artifact outside the checkout, and publishes that exact payload to
  Vulcan after successful branch and tag builds. Pull requests never publish.
- The complete suite passes with `326 passed, 2 skipped`.
- The installed-wheel regression suite passes with `326 passed`.

## Validation commands

From the repository root in the QAT environment:

```bash
pytest -q tests/operators tests/qat tests/integration
pytest -q
git diff --check
```

For the built artifact, install the wheel in a clean environment and repeat at
least the CPU regression suite, import smoke test, and one ONNX export/runtime
case.

## Remaining release gates

1. Run the GitHub workflow and verify its first OIDC-authenticated Vulcan
   publication and generated package metadata.
2. Make wheel construction reproducible without relying on repository Git
   metadata.
3. Publish the public lifecycle contract.
4. Record the supported Python, PyTorch, ONNX, and ONNX Runtime versions in the
   release notes.

The release gate is that QAT behavior is correct, deterministic, documented,
and represented honestly by the manifest.
