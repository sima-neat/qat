---
name: sima-qat-prepare-export
description: Use when preparing PyTorch models with SiMa QAT, finalizing trained QAT graphs, exporting Q/DQ ONNX, validating QAT smoke tests, or updating QAT examples.
---

# Prepare and Export SiMa QAT Models

## Prerequisites

Activate the QAT extension environment first:

```bash
activate-qat
```

If QAT is not installed, use the Neat artifact package for the host architecture:

```bash
sima-cli neat install qat/amd64
sima-cli neat install qat/arm64
```

## Workflow

1. Identify the target `torch.nn.Module` and a representative `example_inputs` tuple.
2. Prefer `sima_qat.prepare(model, example_inputs, target="modalix")`. The
   default auto recipe detects recurrent/state-space modules. Use the low-level
   `sima_prepare_qat_model` only when a framework integration needs individual
   PT2E control points.
3. Run `qat.calibrate(calibration_data)` on representative, deterministically
   ordered data. Omit `batches` to use the recipe-qualified minimum (128 for
   state-space models), then require
   `qat.calibration_report.raise_for_failure()`. Never replace a qualified
   calibration prefix with an arbitrary random subset.
4. Call `qat.freeze()` to lock Model Compiler-compatible power-of-two weight
   scales before the first optimizer update. Training only on the pre-freeze
   grids optimizes a different function and is not target-aware QAT.
5. Train normally. Call `qat.curriculum(step, total_steps)` before the forward,
   then wrap each task loss with
   `qat.loss(task_loss, feature_loss=optional_feature_loss)` to retain matched
   intermediate features and frozen-FP32 behavior. A frozen session remains
   trainable. For a qualified difficult recurrent graph,
   `dropout_probability=` enables training-only elementwise QDrop and
   `dropout_decay_fraction=` must bring it to zero before strict validation.
6. Run `qat.validate(...).raise_for_failure()`.
7. Export with `qat.export(output_directory)`. The bundle contains `model.onnx`
   and a checksum-bound `qat_manifest.json`. CPU export is the default; use
   `export_device` only when a qualification recipe pins it.
8. Audit the ONNX and compiler graph for full Q/DQ/INT8 coverage and compile with the unmodified stock
   Model Compiler. QAT success does not prove that every remaining operator is
   target-realizable.

## Repository Conventions

- Import the recommended customer API from `sima_qat`; keep low-level imports
  in `sima_qat.qat_api` for compatibility integrations.
- Keep generated ONNX outputs under `exported_models/` or the pytest export directory.
- Use CPU smoke coverage unless the behavior specifically requires CUDA.
- Mark quick installation checks with `@pytest.mark.smoke`; mark normal graph coverage with `@pytest.mark.regression`.
- Do not silently skip unsupported quantizer annotations. Add an operator
  regression test or make the model architecture expose supported primitives.

## Validation

Run the narrowest useful command first:

```bash
pytest -m smoke
```

For graph behavior changes, run the affected regression test and then the premerge tox env:

```bash
pytest -m regression
tox -e premerge
```
