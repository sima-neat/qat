# QAT production-readiness TODO

Keep this list scoped to the PyTorch training and standard ONNX QDQ workflow.

## Operator contract

- [x] Maintain one versioned manifest for PyTorch forms, captured ATen forms,
  QAT behavior, expected ONNX operators, and executable test cases.
- [x] Check that every active annotator is represented in the manifest and
  every positive operator case is owned by a manifest entry.
- [x] Cover prepare, real forward/backward, observer freeze, finalization, ONNX
  checking, and representative ONNX Runtime parity.
- [x] Preserve integer outputs for mixed-output operations such as `ArgMax` and
  `TopK`.
- [x] Leave deferred forms trainable and unannotated rather than rejecting the
  whole model or claiming fake-quantized behavior.

## Lifecycle quality

- [x] Remove duplicate BatchNorm graph-rewrite warnings and make repeated
  finalization behavior explicit.
- [x] Resolve the InstanceNorm evaluation export warning and verify inference
  behavior in finalized PyTorch and ONNX Runtime.
- [x] Verify public API input validation, source-model non-mutation, device
  movement, static batch behavior, and opted-in dynamic batch behavior.
- [x] Verify checkpoint round trips before and after `sima_freeze_qat`, including
  exact preservation of the qparams later exported to ONNX.

## Packaging and release

- [x] Build and test the wheel on public GitHub runners before publishing the
  same artifact to Vulcan on branch and tag pushes.
- [ ] Build the package reproducibly without depending on a Git checkout.
- [x] Install the built wheel in a clean CPU environment and run the regression
  suite against the installed package.
- [ ] Run the new workflow in GitHub and verify the first OIDC-authenticated
  Vulcan publication and package metadata.
- [ ] Add lightweight formatting, lint, import, ONNX checker, and documentation
  link checks to CI.
- [ ] Validate and document the supported Python, PyTorch, ONNX, and ONNX
  Runtime versions.
- [ ] Establish semantic versioning, release notes, artifact provenance, and a
  tested tag-to-package promotion path.

## Documentation

- [ ] Keep one concise lifecycle guide for prepare, observer warm-up, freeze,
  recovery training, checkpoint/resume, finalization, and export.
- [ ] Publish the manifest-derived distinction between annotated,
  grid-propagating, mixed-output, decomposed, and pass-through operations.
- [ ] Validate the documented commands against the released wheel.

Production readiness means a clean environment can install the wheel, train and
resume a supported QAT model, freeze and finalize it deterministically, and
export a checker-valid QDQ ONNX model with matching runtime behavior.
