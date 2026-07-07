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
2. Prepare with `sima_prepare_qat_model(model, example_inputs, device)`.
3. Run the normal training or fine-tuning loop on the prepared graph.
4. Finalize with `sima_finalize_qat_model(prepared_model)`.
5. Export with `sima_export_onnx(finalized_model, example_inputs, output_file, device=device)`.

## Repository Conventions

- Import the public API from `sima_qat.qat_api`.
- Keep generated ONNX outputs under `exported_models/` or the pytest export directory.
- Use CPU smoke coverage unless the behavior specifically requires CUDA.
- Mark quick installation checks with `@pytest.mark.smoke`; mark normal graph coverage with `@pytest.mark.regression`.

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
