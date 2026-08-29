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
2. Prepare with `sima_prepare_qat_model(model, example_inputs, device)`. For
   recurrent/state-space models, set `activation_observer="minmax"` and
   `full_range_ste=True` explicitly when the model recipe requires them.
3. Warm up observers in the normal training loop.
4. Call `sima_freeze_qat(prepared_model)` to lock Model
   Compiler-compatible power-of-two weight scales.
5. Fine-tune with frozen observers and fake quantization enabled.
6. Finalize with `sima_finalize_qat_model(prepared_model)`.
7. Export with `sima_export_onnx(finalized_model, example_inputs, output_file,
   device=device)`. CPU export is the default; use `export_device` only when a
   qualification recipe pins it.
8. Audit the ONNX for Q/DQ coverage and compile it with the unmodified stock
   Model Compiler. QAT success does not prove that every remaining operator is
   target-realizable.

## Repository Conventions

- Import the public API from `sima_qat.qat_api`.
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
