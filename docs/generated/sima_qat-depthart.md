# `sima_qat.depthart`

Source: `sima_qat/depthart.py`

One-call DepthART dynamic-p64 QAT setup.

The QAT package deliberately uses a small duck-typed API: it does not import
DepthART or bake customer module paths into the library. A compatible scan
module exposes ``enable_depthart_dynamic_p64_scan_`` and
``freeze_depthart_dynamic_p64_``; the source DepthART SS2D implementation
provides those methods.

## Public API

### Class: `DepthARTP64PreparationReport`

No docstring available.

### Function: `prepare_depthart_dynamic_p64(model, *, profiles, safety_margin, observe)`

Attach exact dynamic-p64 QAT to every compatible DepthART scan.

``profiles`` is keyed by ``named_modules()`` path. Missing entries use
observer-driven base calibration; extra entries fail closed so a stale
profile can never silently bind to a different model revision.

### Function: `freeze_depthart_dynamic_p64(model)`

Freeze every dynamic base and return compiler annotation contracts.

### Function: `enable_depthart_dynamic_p64_fake_quant(model, enabled)`

Switch all prepared scans between float calibration and exact INT8.

### Function: `build_depthart_dynamic_p64_compile_profile(model, onnx_path)`

Bind frozen QAT contracts to exact standard-op ONNX source nodes.

The exported graph contains a real-valued Mul/Add topology shell and one
scalar-one Mul marker for each compiler-exact micro-boundary.  Serial scan
uses ``readout_marker``; associative scan uses ``tree_compose_marker`` and
immediately publishes the result on a normal static INT8 Q/DQ grid.  The
marker's registered initializer encodes the exact ``named_modules()``
path, so this function discovers every occurrence by dataflow instead of
fragile graph order or substring roles.

### Function: `write_depthart_dynamic_p64_compile_profile(model, onnx_path, output_path)`

Build and atomically write the AFE/N2A compile profile.
