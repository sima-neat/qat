# `sima_qat.dynamic_p64`

Source: `sima_qat/dynamic_p64.py`

Compiler-exact dynamic INT8 fake quantization for SiMa's p64 carrier.

The physical value is represented by an inseparable pair::

    value ~= q8 * base * p / 64

``base`` is a calibration-time scalar. ``p`` is selected independently for
each token/group at runtime from ``{64, 128, ..., 16384}`` and is transported
as a single p64 byte (zero is the exact 16384 sentinel).  The implementation
below deliberately mirrors ``RuntimeQKScaleProducer`` in the custom N2A
compiler; it is not generic PyTorch affine fake quantization.

## Public API

### Class: `DynamicP64Result`

Inspectable integer result of :class:`DynamicP64FakeQuant`.

### Class: `StaticInt8Result`

Inspectable result of the dynamic-p64 to static-INT8 boundary.

### Class: `StaticAffineInt8Result`

Inspectable ordinary AFE per-tensor INT8 boundary.

``scale`` is the real step in ``real=(code-zero_point)*scale``.  Keeping
the codes and qparams together prevents the dynamic producer from making
the invalid assumption that an arbitrary static code already uses its
p64 base grid.

### Class: `StaticInt8ToP64FixedPoint`

Safe compiler integer map from an affine INT8 grid to p64's r-grid.

### Class: `P64ToStaticFixedPoint`

Manifest-stable coefficient for a q+p64 readout copy.

The recurrent edge remains dynamic.  Only the copy consumed by ordinary
static-QDQ operators is converted to ``q_static``.

### Class: `P64ProductFixedPoint`

Manifest-stable coefficient for the C128 DepthART product.

### Class: `P64ResidualAddFixedPoint`

Manifest-stable branch alignment for a dynamic q+p64 add.

### Function: `compiler_exact_block_p64_product(lhs, rhs, fixed)`

Exact torch golden for the compiler's DepthART C128 product.

### Function: `compiler_exact_residual_p64_add(lhs, rhs, fixed)`

Exact torch golden for the compiler's compact+compact residual add.

### Function: `compiler_exact_p64_to_static(value, fixed)`

Exact torch golden for the compiler's integer readout boundary.

### Function: `encode_p64(p)`

Encode canonical power-of-two p as signed INT8 p64 bytes.

### Function: `decode_p64(carrier)`

Decode signed/unsigned logical p64 bytes to INT32 p.

### Function: `compiler_exact_dynamic_p64(value, *, base, group_size)`

Apply the N2A p64 producer's exact integer contract.

``group_size=0`` reduces over the complete last axis.  A positive group
size gives one carrier per consecutive group and requires exact division;
padding must be an explicit, architecture-level operation so compiler and
QAT shapes cannot silently disagree.

### Function: `compiler_exact_static_int8_to_p64(value, fixed)`

Mirror the strict-INT8 compiler producer for one affine source grid.

### Class: `StaticAffineInt8FakeQuant`

Per-tensor INT8 Q/DQ immediately before a dynamic-p64 producer.

AFE quantizes ordinary graph edges before the custom recurrence consumes
them.  Modeling only the subsequent dynamic quantizer is therefore too
optimistic: it silently quantizes the original float twice using just the
p64 base.  This module owns the real static grid, exports an ordinary ONNX
Q/DQ pair, and gives :func:`compiler_exact_static_int8_to_p64` the same
codes and affine qparams that the physical kernel receives.

### Class: `DynamicP64FakeQuant`

Straight-through QAT module for the physical q8+p64 program.

Calibration tracks the largest absolute activation and derives
``base = safety_margin * absmax / 32767``.  Call :meth:`freeze_base` before
accuracy qualification or export. Runtime token/group scales remain
dynamic after the base is frozen.

### Class: `P64ToStaticFakeQuant`

STE for the strict-integer q+p64 -> fixed-grid INT8 readout.

This is intentionally separate from :class:`DynamicP64FakeQuant`: the
recurrent stream keeps its runtime p64 carrier, while an ordinary static
INT8 consumer receives only this readout copy.

### Class: `DepthARTDynamicP64Step`

QAT scaffold for one ``state = transition * state + injection`` step.

The forward value is the compiler's integer C128 product and residual-add
program.  Backpropagation uses the derivative of the mathematically
equivalent floating recurrence, avoiding the zero gradient of integer
rounding without pretending the deployed forward is ordinary affine QDQ.

One instance represents one source-proven stack/timestep/block.  Wider
selective-scan states must be split by the model into C128 blocks before
calling this module, exactly like the MLA lowering.
