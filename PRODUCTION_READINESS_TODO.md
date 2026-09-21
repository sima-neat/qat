# QAT production-readiness TODO

This file tracks the work required to release the SiMa QAT package as a
production-supported PyTorch-to-QDQ workflow.

## Ownership boundary

This repository owns:

- PyTorch graph capture and QAT preparation.
- Correct quantization annotations and fake-quantization boundaries.
- Observer and BatchNorm control, grid locking, recovery training, and resume.
- Finalization and standards-compliant ONNX QDQ export.
- Clear validation and diagnostics for supported and unsupported PyTorch forms.

This repository does **not** own graph surgery, AFE compilation, MLA execution,
BoxDecode implementation, or mixed-precision partitioning. Those workflows may
consume QAT output, but they must remain outside this package and its CI.

## 1. Complete operator annotation support

- [ ] Define a versioned operator manifest for the INT8 QAT contract. Every
  supported PyTorch module, functional API, and captured ATen form must map to
  an automated operator case.
- [ ] Classify graph operations correctly: quantizable data-path operations
  receive annotations; layout and shape operations propagate existing grids;
  integer indices and control values preserve their original types; unsupported
  forms fail with a useful error.
- [ ] Complete weighted-operator coverage, including Conv1d, Conv2d, Linear,
  repeated module invocation, supported functional forms, grouped/depthwise
  convolution, and ConvTranspose2d once its PT2E weight-axis contract is fixed.
- [ ] Complete matrix-family coverage for MatMul, MM, BMM, and BAddBMM,
  including supported ranks, broadcasting, constants, and rejected operand
  types.
- [ ] Complete activation and normalization coverage for ReLU, Hardtanh,
  Sigmoid, SiLU, Softmax, LayerNorm, Erf, exact GELU, and their supported fused
  forms. Explicitly reject unsupported approximations such as tanh GELU.
- [ ] Complete elementwise and merge coverage for Add, Mul, Concat, constants,
  repeated inputs, identity padding, fan-out, and shared-grid topologies.
- [ ] Complete pooling and precision-preserving coverage for MaxPool,
  AdaptiveAveragePool, reshape, view, flatten, transpose, permute, squeeze,
  unsqueeze, slice, select, split, chunk, resize/upsample, and padding forms
  used by supported models.
- [ ] Define Embedding/Gather behavior without quantizing integer indices.
  Cover supported table-weight quantization and reject unsupported dynamic
  forms explicitly.
- [ ] Explicitly classify operators outside the W8A8 contract, including
  GridSample, rather than silently inserting misleading annotations.
- [ ] For every supported operator case, test prepare, a real backward/update
  step, observer behavior, freeze, recovery update, finalization, ONNX checker,
  QDQ placement, and ONNX Runtime parity where ONNX Runtime implements the form.
- [ ] Add negative tests for every rejected dtype, rank, axis, operand, dynamic
  weight, and approximation contract.
- [ ] Generate the published operator table from the same manifest used to
  parameterize tests so documentation and implementation cannot drift.

Completion gate: every entry in the versioned manifest has positive and/or
negative lifecycle coverage, and the generated operator table matches the
released wheel.

## 2. Public API hardening

- [ ] Fix the duplicate BatchNorm-node deletion warnings; graph transforms must
  be idempotent and warning-free.
- [ ] Test every public API for input validation, repeated calls, device
  movement, source-model non-mutation, static and opted-in dynamic batch, and
  actionable failures.
- [ ] Make checkpoint/resume behavior explicit and stable across observer-only,
  frozen-grid, and finalized states. Scheduler policy belongs to examples, not
  serialized QAT graph state.
- [ ] Expose a stable, serializable freeze report containing grid retargeting
  and coarsening decisions instead of relying on transient FX metadata.
- [ ] Guarantee that frozen qparams survive state-dict round trips and are the
  exact qparams emitted to ONNX.
- [ ] Define public API compatibility and deprecation policy before releasing
  additional lifecycle functions.

Completion gate: lifecycle and checkpoint behavior are deterministic,
warning-free, documented, and covered through the public API only.

## 3. CI, packaging, and release

- [ ] Make the complete CPU regression suite mandatory for pull requests using
  the built wheel, not the source checkout.
- [ ] Add a CUDA QAT smoke test on a maintained schedule for device movement,
  backward, freeze, finalization, and export. This is a QAT test, not an AFE or
  MLA test.
- [ ] Add formatting, linting, type checking, package import, ONNX checker, and
  documentation-link gates.
- [ ] Replace legacy, git-dependent package construction with a reproducible
  build configuration and separate core runtime dependencies from example and
  test dependencies.
- [ ] Validate the declared Python range against Torch 2.8 and remove any
  unsupported interpreter classifiers or claims.
- [ ] Establish semantic versioning, release notes, artifact provenance, and a
  tested tag-to-package promotion process. A tag build must not skip tests.
- [ ] Set regression coverage thresholds for critical annotation and lifecycle
  code; do not use aggregate line coverage as the only release signal.

Completion gate: a clean environment can install the released wheel and pass
the supported CPU workflow; scheduled CUDA validation passes on the same wheel.

## 4. QAT skill

- [ ] Add a `sima-qat` skill covering environment setup, model assessment, FP32
  baseline collection, QAT preparation, observer warm-up, grid freezing,
  recovery training, finalization, export, and evaluation.
- [ ] Make the skill use the generated operator manifest to explain whether a
  graph is supported and why a form was rejected. It must not implement or
  prescribe graph surgery inside this repository.
- [ ] Include reusable recipes for classification, detection/raw-head export,
  and custom PyTorch training loops without requiring a framework-specific
  trainer.
- [ ] Reference the existing Slurm GPU skill for cluster execution rather than
  duplicating cluster-specific instructions.
- [ ] Add failure playbooks for capture, annotation, scale locking, accuracy
  loss, checkpoint/resume, finalization, and ONNX export.
- [ ] Validate the skill from a clean environment on one classification model
  and one detection model.

Completion gate: a user or agent can produce and evaluate a supported QDQ model
without undocumented repository knowledge.

## 5. Main `core` documentation integration

- [ ] Add QAT to `core/docs` and link it from the model-development and model
  compiler documentation.
- [ ] Document installation, supported versions, the public lifecycle, training
  ownership, BatchNorm policy, observer warm-up, grid locking, recovery, and
  export.
- [ ] Publish the generated operator manifest and explain the difference
  between annotated operators, grid-propagating operators, type-preserving
  operators, and explicitly unsupported forms.
- [ ] Add an end-to-end tutorial measuring FP32, pre-training fake INT8, trained
  QAT, finalized PyTorch, and ONNX Runtime results with identical preprocessing
  and evaluation.
- [ ] Document detection ownership clearly: QAT exports model tensors or raw
  heads; deployment postprocessing and BoxDecode remain downstream concerns.
- [ ] Add reproducibility, checkpoint/resume, diagnostics, troubleshooting, and
  release-compatibility guidance.
- [ ] Link the QAT README to the canonical `core` pages and link the canonical
  pages back to maintained examples and the exact supported package release.

Completion gate: QAT is discoverable from the main documentation, its ownership
and support boundaries are explicit, and all documented commands are validated
against the released wheel.

