# QAT production-readiness TODO

This file tracks the remaining work required before the SiMa QAT package can be
presented as a production-supported workflow. A checked item must have automated
coverage and user-facing documentation; implementation alone is not sufficient.

## 1. Complete operator support

- [ ] Generate a source-of-truth operator matrix from the current
  `awesome-frontend` contracts. Classify every operator and form as supported,
  mixed precision, graph-surgery required, or unsupported.
- [ ] Close the remaining INT8 annotation gaps for AFE-supported operators and
  their common PyTorch forms. Do not equate `any_shape_on_mla` with universal
  rank, axis, data-type, or operand support.
- [ ] Resolve the ConvTranspose2d weight-grid/export contract with AFE before
  advertising it as supported.
- [ ] Define and implement the supported Embedding/Gather policy. Preserve
  integer indices and verify signed INT8 table export and compiler ingestion.
- [ ] Keep GridSample out of the INT8 annotation set while AFE supports it only
  in BF16; document the mixed-precision or graph-surgery path instead.
- [ ] Complete coverage for operator reuse and graph topology edge cases,
  including repeated module invocation, repeated concat inputs, shared grids,
  constants, identity padding, and dynamic/static operand combinations.
- [ ] For every supported form, test the complete lifecycle:
  prepare, observer warm-up, freeze, recovery training, finalization, ONNX QDQ
  export, AFE compilation, and MLA output parity.
- [ ] Add negative tests with actionable diagnostics for every rejected form.
- [ ] Publish the tested operator matrix, including shape, axis, precision,
  dynamic-input, and graph-surgery constraints.

Completion gate: the published matrix and automated matrix must agree, and all
advertised forms must compile and execute on MLA in CI.

## 2. QAT skills

- [ ] Add a `sima-qat` skill that guides an agent through environment setup,
  model assessment, FP32 baseline collection, QAT preparation, calibration,
  grid freezing, recovery training, finalization, export, and evaluation.
- [ ] Route operator-compatibility questions through the AFE operator matrix and
  route unsupported graphs through the existing model-surgery workflow.
- [ ] Include reusable recipes for classification, detection/BoxDecode, and
  custom training loops without requiring a framework-specific trainer.
- [ ] Include Slurm GPU submission, monitoring, resume, and artifact-validation
  guidance without duplicating the cluster-specific Slurm skill.
- [ ] Add failure playbooks for capture errors, unsupported annotations,
  scale-locking failures, accuracy loss, ONNX export failures, and AFE compiler
  rejection.
- [ ] Validate the skill against at least one clean classification model and one
  detection model from baseline through compiled artifact.

Completion gate: a user or agent starting from a supported PyTorch checkpoint
can produce and validate an MLA-compatible QDQ model by following the skill
without undocumented internal knowledge.

## 3. Main `core` documentation integration

- [ ] Add QAT to the main documentation under `core/docs`, including navigation
  from the model-compiler and model-development documentation.
- [ ] Document installation, supported PyTorch/Python versions, and the public
  API lifecycle: prepare, optional BatchNorm freeze, observer warm-up, QAT-grid
  freeze, recovery training, finalization, and ONNX export.
- [ ] Publish the operator matrix and clearly distinguish native INT8 support,
  BF16/mixed-precision support, automatic graph transformations, required graph
  surgery, and unsupported forms.
- [ ] Add an end-to-end tutorial with FP32, pre-QAT fake-INT8, trained QAT, ONNX,
  AFE, and MLA parity/accuracy measurements.
- [ ] Document BoxDecode integration and ownership boundaries: the QAT model
  exports raw heads, while deployment postprocessing follows the supported
  BoxDecode contract.
- [ ] Add checkpoint/resume, reproducibility, troubleshooting, and production
  sign-off guidance.
- [ ] Link the QAT repository README to the canonical `core` pages and link the
  canonical pages back to maintained examples and the exact supported release.
- [ ] Add documentation build/link checks and ensure examples use commands that
  run in the released environment.

Completion gate: QAT is discoverable from the main SiMa documentation, its
support boundaries are explicit, and every documented workflow is exercised by
release validation.

