# Inspect frozen QAT exports without changing their contract

An executable ONNX numerical reference is not necessarily a Model Compiler
input. Native-aware reference graphs can contain exact diagnostic arithmetic,
validity guards and explicit nondeployment metadata.

`sima_qat/export_validation.py` checks those declarations without training,
rewriting ONNX, importing a compiler or touching a device. It is also a standalone
CPU tool with NumPy and ONNX, so inspecting a file does not require importing
the training API:

```bash
python sima_qat/export_validation.py inspect --onnx model.onnx \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" --require-compiler-input
```

The tool verifies structural validity, records port types/shapes and operation
counts, rejects duplicate metadata, checks the checkpoint identity and reports
explicit eligibility exclusions. The strict command exits 2 for a diagnostic
graph, missing eligibility, a checkpoint mismatch or a public boolean guard.
Without `--require-compiler-input`, inspection itself can succeed while the
printed `compiler_input_preflight` remains false.

A positive declaration is **only preflight**, never proof of target support,
preserved precision, compiled parity or accuracy. Do not change diagnostic
metadata to get past a gate; produce and independently qualify the correct
artifact instead.

Compare two existing trusted output captures separately:

```bash
python sima_qat/export_validation.py compare \
  --expected pytorch-outputs.npz --actual onnx-outputs.npz
```

The comparison requires identical nonempty output sets, matching shapes/dtypes,
finite values and true boolean validity guards. It compares element bit patterns,
including signed zero, and records both file checksums. Exit 0 means exact
captured outputs, 1 means a measured difference, and 2 means invalid/incomplete
evidence. This command runs neither model; a saved-array comparison must not be
described as fresh PyTorch, compiler, hardware or dataset evaluation.
