"""Compiler-exact dynamic INT8 fake quantization for SiMa's p64 carrier.

The physical value is represented by an inseparable pair::

    value ~= q8 * base * p / 64

``base`` is a calibration-time scalar. ``p`` is selected independently for
each token/group at runtime from ``{64, 128, ..., 16384}`` and is transported
as a single p64 byte (zero is the exact 16384 sentinel).  The implementation
below deliberately mirrors ``RuntimeQKScaleProducer`` in the custom N2A
compiler; it is not generic PyTorch affine fake quantization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn

INT16_LIMIT: Final[int] = 32767
M_FLOOR: Final[int] = 128
POW2_K: Final[int] = 22
P64_MIN: Final[int] = 64
P64_MAX: Final[int] = 16384


@dataclass(frozen=True)
class DynamicP64Result:
    """Inspectable integer result of :class:`DynamicP64FakeQuant`."""

    codes: Tensor
    p: Tensor
    carrier: Tensor
    dequantized: Tensor
    clipped_r_count: int


@dataclass(frozen=True)
class StaticInt8Result:
    """Inspectable result of the dynamic-p64 to static-INT8 boundary."""

    codes: Tensor
    dequantized: Tensor
    clipped_count: int


@dataclass(frozen=True)
class StaticAffineInt8Result:
    """Inspectable ordinary AFE per-tensor INT8 boundary.

    ``scale`` is the real step in ``real=(code-zero_point)*scale``.  Keeping
    the codes and qparams together prevents the dynamic producer from making
    the invalid assumption that an arbitrary static code already uses its
    p64 base grid.
    """

    codes: Tensor
    dequantized: Tensor
    scale: float
    zero_point: int
    clipped_count: int


@dataclass(frozen=True)
class StaticInt8ToP64FixedPoint:
    """Safe compiler integer map from an affine INT8 grid to p64's r-grid."""

    input_scale: float
    output_base: float
    zero_point: int
    multiplier: int
    shift: int
    max_relative_error: float
    int32_bound: int

    @classmethod
    def derive(
        cls,
        input_scale: float,
        output_base: float,
        *,
        zero_point: int = 0,
        max_shift: int = 30,
        max_relative_error: float = 5.0e-3,
    ) -> StaticInt8ToP64FixedPoint:
        input_scale = _require_positive_finite("input_scale", input_scale)
        output_base = _require_positive_finite("output_base", output_base)
        zero_point = int(zero_point)
        if not -128 <= zero_point <= 127:
            raise ValueError(f"zero_point is outside INT8: {zero_point}")
        coefficient = input_scale / output_base
        centered_bound = max(abs(-128 - zero_point), abs(127 - zero_point))
        int32_max = (1 << 31) - 1
        multiplier_limit = int32_max // max(centered_bound, 1)
        best = None
        for shift in range(int(max_shift), -1, -1):
            multiplier = round(coefficient * float(1 << shift))
            if not 0 < multiplier <= multiplier_limit:
                continue
            approximation = multiplier / float(1 << shift)
            error = abs(approximation - coefficient) / coefficient
            if error <= max_relative_error:
                best = multiplier, shift, error
                break
        if best is None:
            raise ValueError(
                "static-INT8 to p64 coefficient has no safe INT32 form: "
                f"coefficient={coefficient} centered_bound={centered_bound}")
        multiplier, shift, error = best
        bound = centered_bound * multiplier
        if bound > int32_max:
            raise AssertionError(f"static-to-p64 INT32 proof overflow: {bound}")
        return cls(
            input_scale, output_base, zero_point, multiplier, shift, error,
            bound)


@dataclass(frozen=True)
class P64ToStaticFixedPoint:
    """Manifest-stable coefficient for a q+p64 readout copy.

    The recurrent edge remains dynamic.  Only the copy consumed by ordinary
    static-QDQ operators is converted to ``q_static``.
    """

    input_base: float
    output_scale: float
    multiplier: int
    shift: int
    max_relative_error: float
    int32_bound: int

    @classmethod
    def derive(
        cls,
        input_base: float,
        output_scale: float,
        *,
        max_shift: int = 30,
        max_relative_error: float = 5.0e-3,
    ) -> P64ToStaticFixedPoint:
        input_base = _require_positive_finite("input_base", input_base)
        output_scale = _require_positive_finite("output_scale", output_scale)
        coefficient = input_base / (64.0 * output_scale)
        qp_bound = 127 * P64_MAX
        int32_max = (1 << 31) - 1
        multiplier_limit = int32_max // qp_bound
        best = None
        for shift in range(int(max_shift), -1, -1):
            multiplier = round(coefficient * float(1 << shift))
            if not 0 < multiplier <= multiplier_limit:
                continue
            approximation = multiplier / float(1 << shift)
            error = abs(approximation - coefficient) / coefficient
            if error <= max_relative_error:
                best = multiplier, shift, error
                break
        if best is None:
            raise ValueError(
                "DepthART p64-to-static coefficient has no safe INT32 form: "
                f"coefficient={coefficient}")
        multiplier, shift, error = best
        bound = qp_bound * multiplier
        if bound > int32_max:
            raise AssertionError(
                f"DepthART p64-to-static INT32 bound overflow: {bound}")
        return cls(
            input_base, output_scale, multiplier, shift, error, bound)


@dataclass(frozen=True)
class P64ProductFixedPoint:
    """Manifest-stable coefficient for the C128 DepthART product."""

    lhs_base: float
    rhs_base: float
    output_base: float
    multiplier: int
    shift: int
    pre_shift: int
    max_relative_error: float
    int32_bound: int

    @classmethod
    def derive(
        cls,
        lhs_base: float,
        rhs_base: float,
        output_base: float,
        *,
        max_shift: int = 30,
        max_pre_shift: int = 16,
        max_relative_error: float = 5.0e-3,
    ) -> P64ProductFixedPoint:
        lhs_base = _require_positive_finite("lhs_base", lhs_base)
        rhs_base = _require_positive_finite("rhs_base", rhs_base)
        output_base = _require_positive_finite("output_base", output_base)
        coefficient = lhs_base * rhs_base / output_base
        exact_bound = (127 * (P64_MAX // P64_MIN)) ** 2
        int32_max = (1 << 31) - 1
        best = None
        for pre_shift in range(int(max_pre_shift) + 1):
            reduced_bound = (exact_bound + (1 << pre_shift) - 1) >> pre_shift
            multiplier_limit = int32_max // reduced_bound
            for shift in range(int(max_shift), -1, -1):
                multiplier = round(coefficient * float(1 << (shift + pre_shift)))
                if not 0 < multiplier <= multiplier_limit:
                    continue
                approximation = multiplier / float(1 << (shift + pre_shift))
                error = abs(approximation - coefficient) / coefficient
                if error <= max_relative_error:
                    best = multiplier, shift, pre_shift, error, reduced_bound
                    break
            if best is not None:
                break
        if best is None:
            raise ValueError(
                "DepthART p64 product coefficient has no safe INT32 form: "
                f"coefficient={coefficient}")
        multiplier, shift, pre_shift, error, reduced_bound = best
        bound = reduced_bound * multiplier
        return cls(
            lhs_base, rhs_base, output_base, multiplier, shift, pre_shift,
            error, bound)


@dataclass(frozen=True)
class P64ResidualAddFixedPoint:
    """Manifest-stable branch alignment for a dynamic q+p64 add."""

    lhs_shift: int
    rhs_shift: int
    lhs_multiplier: int
    rhs_multiplier: int
    lhs_base: float
    rhs_base: float
    output_base: float
    max_coefficient_relative_error: float

    @classmethod
    def derive(
        cls,
        lhs_base: float,
        rhs_base: float,
        output_base: float,
        *,
        max_shift: int = 20,
        max_relative_error: float = 5.0e-3,
    ) -> P64ResidualAddFixedPoint:
        lhs_base = _require_positive_finite("lhs_base", lhs_base)
        rhs_base = _require_positive_finite("rhs_base", rhs_base)
        output_base = _require_positive_finite("output_base", output_base)
        qp_bound = 127 * P64_MAX
        product_limit = (1 << 31) - 1

        def derive_one(coefficient: float) -> tuple[int, int, float]:
            safe = math.floor(math.log2(
                product_limit / (qp_bound * coefficient)))
            shift = max(0, min(int(max_shift), safe))
            multiplier = round(coefficient * float(1 << shift))
            while multiplier > 0 and qp_bound * multiplier > product_limit and shift > 0:
                shift -= 1
                multiplier = round(coefficient * float(1 << shift))
            if multiplier <= 0 or qp_bound * multiplier > product_limit:
                raise ValueError(
                    "DepthART residual coefficient has no safe INT32 Q-format")
            approximation = multiplier / float(1 << shift)
            error = abs(approximation - coefficient) / coefficient
            return shift, multiplier, error

        lhs_coefficient = lhs_base / (64.0 * output_base)
        rhs_coefficient = rhs_base / (64.0 * output_base)
        lhs_shift, lhs_multiplier, lhs_error = derive_one(lhs_coefficient)
        rhs_shift, rhs_multiplier, rhs_error = derive_one(rhs_coefficient)
        error = max(lhs_error, rhs_error)
        if error > max_relative_error:
            raise ValueError(
                f"DepthART residual coefficient error {error:.6g} exceeds "
                f"{max_relative_error}")
        aligned_bound = (
            math.ceil(qp_bound * lhs_multiplier / float(1 << lhs_shift))
            + math.ceil(qp_bound * rhs_multiplier / float(1 << rhs_shift)))
        if aligned_bound > product_limit:
            raise ValueError("DepthART residual aligned sum can overflow INT32")
        return cls(
            lhs_shift, rhs_shift, lhs_multiplier, rhs_multiplier,
            lhs_base, rhs_base, output_base, error)


def _require_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, found {value}")
    return value


def _round_half_away(value: Tensor) -> Tensor:
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def _exact_forward_ste(exact: Tensor, surrogate: Tensor) -> Tensor:
    """Return ``exact`` bit-for-bit while differentiating as ``surrogate``.

    ``surrogate + (exact - surrogate).detach()`` is algebraically equivalent
    but performs a cancellation in the deployed forward and can perturb a
    float32 QDQ lattice point by one ulp.  Subtracting a tensor from its own
    detached view produces exact zero, so this form preserves both the exact
    integer oracle forward and the intended straight-through gradient.
    """
    return exact.detach() + (surrogate - surrogate.detach())


def _ceil_log2_idx(idx: Tensor) -> Tensor:
    """Exact integer ceil(log2(idx)) for idx in [1, 255]."""

    result = torch.zeros_like(idx, dtype=torch.int64)
    # Count the powers of two strictly below idx.  This preserves exact-power
    # boundaries without relying on a floating log implementation.
    for threshold in (1, 2, 4, 8, 16, 32, 64, 128):
        result = result + (idx > threshold).to(torch.int64)
    return result


def _round_even_shift(value: Tensor, shift: int) -> Tensor:
    """Signed integer round-to-nearest-even division by ``2**shift``."""

    shift = int(shift)
    value = value.to(torch.int64)
    if shift == 0:
        return value
    base = torch.bitwise_right_shift(value, shift)
    remainder = value - torch.bitwise_left_shift(base, shift)
    half = 1 << (shift - 1)
    increment = (remainder > half) | (
        (remainder == half) & (torch.bitwise_and(base, 1) != 0))
    return base + increment.to(torch.int64)


def _round_half_away_shift(value: Tensor, shift: int) -> Tensor:
    """Signed integer nearest rounding with ties away from zero."""

    shift = int(shift)
    value = value.to(torch.int64)
    if shift < 0:
        raise ValueError("shift must be non-negative")
    if shift == 0:
        return value
    half = 1 << (shift - 1)
    sign = torch.where(value < 0, -torch.ones_like(value), torch.zeros_like(value))
    return torch.bitwise_right_shift(value + half + sign, shift)


def _row_p(result: DynamicP64Result) -> Tensor:
    """Normalize the one-group carrier to ``q.shape[:-1] + (1,)``."""

    rows = math.prod(result.codes.shape[:-1])
    if result.p.numel() != rows:
        raise ValueError(
            "DepthART C128 arithmetic requires exactly one p64 carrier per row")
    return result.p.to(torch.int64).reshape(*result.codes.shape[:-1], 1)


def _narrow_p64(r_unclipped: Tensor, *, output_base: float, dtype: torch.dtype) -> DynamicP64Result:
    """Compiler-exact INT16 clip and power-of-two q8+p64 tail."""

    clipped_count = int(torch.count_nonzero(
        torch.abs(r_unclipped) > INT16_LIMIT).item())
    r = torch.clamp(r_unclipped.to(torch.int64), -INT16_LIMIT, INT16_LIMIT)
    row_max = torch.clamp(torch.amax(torch.abs(r), dim=-1, keepdim=True), min=M_FLOOR)
    idx = torch.clamp(torch.bitwise_right_shift(row_max, 7), 1, 255)
    exponent = 7 + _ceil_log2_idx(idx)
    p = torch.bitwise_left_shift(
        torch.ones_like(exponent, dtype=torch.int64), exponent - 1)
    multiplier = torch.bitwise_left_shift(
        torch.ones_like(exponent, dtype=torch.int64), POW2_K - exponent)
    q = torch.clamp(
        torch.bitwise_right_shift(r * multiplier + (1 << 14), 15),
        -127, 127).to(torch.int8)
    dequantized = (
        q.to(torch.float64) * float(output_base) * p.to(torch.float64) / 64.0
    ).to(dtype)
    return DynamicP64Result(
        codes=q,
        p=p.to(torch.int32),
        carrier=encode_p64(p),
        dequantized=dequantized,
        clipped_r_count=clipped_count,
    )


def compiler_exact_block_p64_product(
    lhs: DynamicP64Result,
    rhs: DynamicP64Result,
    fixed: P64ProductFixedPoint,
) -> DynamicP64Result:
    """Exact torch golden for the compiler's DepthART C128 product."""

    if lhs.codes.shape != rhs.codes.shape or int(lhs.codes.shape[-1]) != 128:
        raise ValueError("DepthART p64 product requires matching C128 code tensors")
    lhs_p = _row_p(lhs)
    rhs_p = _row_p(rhs)
    lhs_expanded = lhs.codes.to(torch.int64) * torch.bitwise_right_shift(lhs_p, 6)
    rhs_expanded = rhs.codes.to(torch.int64) * torch.bitwise_right_shift(rhs_p, 6)
    exact = lhs_expanded * rhs_expanded
    reduced = _round_even_shift(exact, fixed.pre_shift)
    wide = reduced * int(fixed.multiplier)
    if bool(torch.any(torch.abs(wide) > (1 << 31) - 1)):
        raise OverflowError("DepthART p64 product exceeded its INT32 proof")
    r = _round_even_shift(wide, fixed.shift)
    return _narrow_p64(
        r, output_base=fixed.output_base, dtype=lhs.dequantized.dtype)


def compiler_exact_residual_p64_add(
    lhs: DynamicP64Result,
    rhs: DynamicP64Result,
    fixed: P64ResidualAddFixedPoint,
) -> DynamicP64Result:
    """Exact torch golden for the compiler's compact+compact residual add."""

    if lhs.codes.shape != rhs.codes.shape:
        raise ValueError("DepthART residual q tensors must have matching shapes")
    lhs_aligned = _round_even_shift(
        lhs.codes.to(torch.int64) * _row_p(lhs) * int(fixed.lhs_multiplier),
        fixed.lhs_shift)
    rhs_aligned = _round_even_shift(
        rhs.codes.to(torch.int64) * _row_p(rhs) * int(fixed.rhs_multiplier),
        fixed.rhs_shift)
    return _narrow_p64(
        lhs_aligned + rhs_aligned,
        output_base=fixed.output_base,
        dtype=lhs.dequantized.dtype)


def compiler_exact_p64_to_static(
    value: DynamicP64Result,
    fixed: P64ToStaticFixedPoint,
) -> StaticInt8Result:
    """Exact torch golden for the compiler's integer readout boundary."""

    p = _row_p(value)
    wide = value.codes.to(torch.int64) * p * int(fixed.multiplier)
    if bool(torch.any(torch.abs(wide) > (1 << 31) - 1)):
        raise OverflowError("DepthART p64-to-static exceeded its INT32 proof")
    aligned = _round_even_shift(wide, fixed.shift)
    clipped_count = int(torch.count_nonzero(torch.abs(aligned) > 127).item())
    codes = torch.clamp(aligned, -127, 127).to(torch.int8)
    dequantized = (
        codes.to(torch.float64) * float(fixed.output_scale)
    ).to(value.dequantized.dtype)
    return StaticInt8Result(codes, dequantized, clipped_count)


def encode_p64(p: Tensor) -> Tensor:
    """Encode canonical power-of-two p as signed INT8 p64 bytes."""

    p64 = p.to(torch.int64)
    valid_range = (p64 >= P64_MIN) & (p64 <= P64_MAX)
    valid_power = (p64 & (p64 - 1)) == 0
    if not bool(torch.all(valid_range & valid_power)):
        raise ValueError("p contains a noncanonical p64 value")
    # PyTorch has no unsigned-byte view equivalent to NumPy's.  The physical
    # byte 0x80 is represented by signed INT8 -128 and 0x00 is the sentinel.
    code_u8 = torch.bitwise_and(torch.bitwise_right_shift(p64, 6), 0xFF)
    code_i16 = torch.where(code_u8 >= 128, code_u8 - 256, code_u8)
    return code_i16.to(torch.int8)


def decode_p64(carrier: Tensor) -> Tensor:
    """Decode signed/unsigned logical p64 bytes to INT32 p."""

    if carrier.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"p64 carrier must be INT8/UINT8, found {carrier.dtype}")
    code = carrier.to(torch.int16)
    if carrier.dtype == torch.int8:
        code = torch.where(code < 0, code + 256, code)
    p = torch.where(code == 0, torch.full_like(code, P64_MAX), code << 6)
    valid = (p >= P64_MIN) & (p <= P64_MAX) & ((p & (p - 1)) == 0)
    if not bool(torch.all(valid)):
        raise ValueError("carrier contains a noncanonical p64 code")
    return p.to(torch.int32)


def compiler_exact_dynamic_p64(
    value: Tensor,
    *,
    base: Tensor | float,
    group_size: int = 0,
) -> DynamicP64Result:
    """Apply the N2A p64 producer's exact integer contract.

    ``group_size=0`` reduces over the complete last axis.  A positive group
    size gives one carrier per consecutive group and requires exact division;
    padding must be an explicit, architecture-level operation so compiler and
    QAT shapes cannot silently disagree.
    """

    if not isinstance(value, Tensor) or value.ndim < 2 or value.shape[-1] < 1:
        raise ValueError(f"value requires token axes plus channels, got {value}")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("value contains NaN or infinity")
    width = int(value.shape[-1])
    group_size = width if int(group_size) == 0 else int(group_size)
    if group_size < 1 or width % group_size:
        raise ValueError(
            f"group_size {group_size} must divide the last-axis width {width}"
        )

    # The compiler persists the Python/manifest base as a binary64 scalar,
    # uses only its reciprocal as float32 for the r-grid, and constructs the
    # diagnostic/dequant scale in binary64. Preserve that split exactly.
    base_tensor = torch.as_tensor(base, dtype=torch.float64, device=value.device)
    if base_tensor.numel() != 1:
        raise ValueError("dynamic p64 currently requires one static base per boundary")
    if not bool(torch.isfinite(base_tensor).all()) or not bool((base_tensor > 0).all()):
        raise ValueError("base must be finite and positive")

    original_shape = value.shape
    grouped = value.to(torch.float32).reshape(
        *original_shape[:-1], width // group_size, group_size
    )
    # Match RuntimeQKScaleProducer: float32 reciprocal and product, followed by
    # half-away rounding on the INT16 calibration grid.
    inverse = torch.reciprocal(base_tensor).to(torch.float32)
    rounded = _round_half_away((grouped * inverse).to(torch.float64))
    clipped_r_count = int(torch.count_nonzero(torch.abs(rounded) > INT16_LIMIT).item())
    r = torch.clamp(rounded, -INT16_LIMIT, INT16_LIMIT).to(torch.int64)

    row_max = torch.amax(torch.abs(r), dim=-1, keepdim=True)
    row_max = torch.clamp(row_max, min=M_FLOOR)
    idx = torch.clamp(torch.bitwise_right_shift(row_max, 7), 1, 255)
    exponent = 7 + _ceil_log2_idx(idx)
    p = torch.bitwise_left_shift(
        torch.ones_like(exponent, dtype=torch.int64), exponent - 1
    )
    multiplier = torch.bitwise_left_shift(
        torch.ones_like(exponent, dtype=torch.int64), POW2_K - exponent
    )

    # MLA POW2 narrow is floor(x / 2**15 + .5) for signed integers. Arithmetic
    # right shift after adding 2**14 has exactly those semantics.
    q = torch.bitwise_right_shift(r * multiplier + (1 << 14), 15)
    q = torch.clamp(q, -127, 127).to(torch.int8)
    dequant_scale = base_tensor.to(torch.float64) * p.to(torch.float64) / 64.0
    dequantized = (q.to(torch.float64) * dequant_scale).to(value.dtype)
    return DynamicP64Result(
        codes=q.reshape(original_shape),
        p=p.to(torch.int32),
        carrier=encode_p64(p),
        dequantized=dequantized.reshape(original_shape),
        clipped_r_count=clipped_r_count,
    )


def compiler_exact_static_int8_to_p64(
    value: StaticAffineInt8Result,
    fixed: StaticInt8ToP64FixedPoint,
) -> DynamicP64Result:
    """Mirror the strict-INT8 compiler producer for one affine source grid."""

    if value.codes.dtype != torch.int8 or value.codes.ndim < 2:
        raise ValueError(
            "static-to-p64 requires an INT8 tensor with token/channel axes")
    if int(value.codes.shape[-1]) != 128:
        raise ValueError(
            f"DepthART static-to-p64 requires C128, got {value.codes.shape}")
    if value.scale != fixed.input_scale or value.zero_point != fixed.zero_point:
        raise ValueError(
            "static-to-p64 qparams do not match the fixed compiler contract")
    centered = value.codes.to(torch.int64) - int(fixed.zero_point)
    wide = centered * int(fixed.multiplier)
    if bool(torch.any(torch.abs(wide) > (1 << 31) - 1)):
        raise OverflowError("static-to-p64 fixed multiply exceeded INT32 proof")
    r = _round_half_away_shift(wide, fixed.shift)
    return _narrow_p64(
        r, output_base=fixed.output_base, dtype=value.dequantized.dtype)


class StaticAffineInt8FakeQuant(nn.Module):
    """Per-tensor INT8 Q/DQ immediately before a dynamic-p64 producer.

    AFE quantizes ordinary graph edges before the custom recurrence consumes
    them.  Modeling only the subsequent dynamic quantizer is therefore too
    optimistic: it silently quantizes the original float twice using just the
    p64 base.  This module owns the real static grid, exports an ordinary ONNX
    Q/DQ pair, and gives :func:`compiler_exact_static_int8_to_p64` the same
    codes and affine qparams that the physical kernel receives.
    """

    qmin = -128
    qmax = 127

    def __init__(
        self,
        *,
        scale: float = 1.0 / 127.0,
        zero_point: int = 0,
        safety_margin: float = 1.0,
        observe: bool = True,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.symmetric = bool(symmetric)
        self.safety_margin = _require_positive_finite(
            "safety_margin", safety_margin)
        zero_point = int(zero_point)
        if not self.qmin <= zero_point <= self.qmax:
            raise ValueError(f"zero_point is outside INT8: {zero_point}")
        self.register_buffer(
            "scale", torch.tensor(
                _require_positive_finite("scale", scale), dtype=torch.float64))
        self.register_buffer(
            "zero_point", torch.tensor(zero_point, dtype=torch.int32))
        self.register_buffer(
            "observed_min", torch.tensor(float("inf"), dtype=torch.float64))
        self.register_buffer(
            "observed_max", torch.tensor(float("-inf"), dtype=torch.float64))
        self.register_buffer(
            "observer_enabled", torch.tensor(bool(observe), dtype=torch.bool),
            persistent=False)
        self.register_buffer(
            "fake_quant_enabled", torch.tensor(True, dtype=torch.bool),
            persistent=False)

    @torch.no_grad()
    def _observe(self, value: Tensor) -> None:
        finite = value.detach().to(torch.float32).to(torch.float64)
        if not bool(torch.all(torch.isfinite(finite))):
            raise ValueError("static INT8 observer received NaN or infinity")
        self.observed_min.copy_(torch.minimum(self.observed_min, torch.amin(finite)))
        self.observed_max.copy_(torch.maximum(self.observed_max, torch.amax(finite)))
        lower = torch.minimum(self.observed_min, torch.zeros_like(self.observed_min))
        upper = torch.maximum(self.observed_max, torch.zeros_like(self.observed_max))
        minimum = torch.finfo(torch.float64).tiny
        if self.symmetric:
            scale = torch.clamp(
                torch.maximum(torch.abs(lower), torch.abs(upper))
                * self.safety_margin / 127.0,
                min=minimum)
            zero_point = torch.zeros((), dtype=torch.int32, device=value.device)
        else:
            span = (upper - lower) * self.safety_margin
            scale = torch.clamp(
                span / float(self.qmax - self.qmin), min=minimum)
            zero_point = torch.round(self.qmin - lower / scale).to(torch.int32)
            zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

    @torch.no_grad()
    def freeze_qparams(self) -> StaticAffineInt8FakeQuant:
        self.observer_enabled.fill_(False)
        return self

    @torch.no_grad()
    def enable_fake_quant(self, enabled: bool = True) -> StaticAffineInt8FakeQuant:
        self.fake_quant_enabled.fill_(bool(enabled))
        return self

    def quantize(self, value: Tensor) -> StaticAffineInt8Result:
        scale = float(self.scale)
        zero_point = int(self.zero_point)
        scaled = value.to(torch.float64) / scale
        unclipped = torch.round(scaled) + zero_point
        clipped_count = int(torch.count_nonzero(
            (unclipped < self.qmin) | (unclipped > self.qmax)).item())
        codes = torch.clamp(unclipped, self.qmin, self.qmax).to(torch.int8)
        dequantized = (
            (codes.to(torch.float64) - zero_point) * scale).to(value.dtype)
        return StaticAffineInt8Result(
            codes, dequantized, scale, zero_point, clipped_count)

    def forward(self, value: Tensor) -> Tensor:
        if bool(self.observer_enabled):
            self._observe(value)
        if not bool(self.fake_quant_enabled):
            return value
        if torch.onnx.is_in_onnx_export():
            # Legacy and dynamo ONNX exporters both recognize this public op
            # and emit a standard QuantizeLinear/DequantizeLinear boundary.
            return torch.fake_quantize_per_tensor_affine(
                value, float(self.scale), int(self.zero_point),
                self.qmin, self.qmax)
        exact = self.quantize(value).dequantized
        return _exact_forward_ste(exact, value)


class DynamicP64FakeQuant(nn.Module):
    """Straight-through QAT module for the physical q8+p64 program.

    Calibration tracks the largest absolute activation and derives
    ``base = safety_margin * absmax / 32767``.  Call :meth:`freeze_base` before
    accuracy qualification or export. Runtime token/group scales remain
    dynamic after the base is frozen.
    """

    def __init__(
        self,
        *,
        base: float = 1.0 / INT16_LIMIT,
        group_size: int = 0,
        safety_margin: float = 1.0,
        observe: bool = True,
    ) -> None:
        super().__init__()
        self.group_size = int(group_size)
        if self.group_size < 0:
            raise ValueError("group_size must be non-negative")
        self.safety_margin = _require_positive_finite("safety_margin", safety_margin)
        self.register_buffer(
            "base", torch.tensor(_require_positive_finite("base", base), dtype=torch.float64)
        )
        self.register_buffer("observed_absmax", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer(
            "observer_enabled", torch.tensor(bool(observe), dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "fake_quant_enabled", torch.tensor(True, dtype=torch.bool),
            persistent=False,
        )

    @torch.no_grad()
    def enable_observer(self, enabled: bool = True) -> DynamicP64FakeQuant:
        self.observer_enabled.fill_(bool(enabled))
        return self

    @torch.no_grad()
    def freeze_base(self) -> DynamicP64FakeQuant:
        self.observer_enabled.fill_(False)
        return self

    @torch.no_grad()
    def enable_fake_quant(self, enabled: bool = True) -> DynamicP64FakeQuant:
        self.fake_quant_enabled.fill_(bool(enabled))
        return self

    @torch.no_grad()
    def _observe(self, value: Tensor) -> None:
        # Inputs are compiler-calibration float32 values. Promote those exact
        # values before applying the manifest safety margin in binary64.
        finite = value.detach().to(torch.float32).to(torch.float64)
        if not bool(torch.all(torch.isfinite(finite))):
            raise ValueError("dynamic p64 observer received NaN or infinity")
        current = torch.amax(torch.abs(finite))
        self.observed_absmax.copy_(torch.maximum(self.observed_absmax, current))
        effective = torch.clamp(
            self.observed_absmax * self.safety_margin, min=torch.finfo(torch.float64).tiny
        )
        self.base.copy_(effective / float(INT16_LIMIT))

    def quantize(self, value: Tensor) -> DynamicP64Result:
        return compiler_exact_dynamic_p64(
            value, base=self.base, group_size=self.group_size
        )

    def forward(self, value: Tensor) -> Tensor:
        if bool(self.observer_enabled):
            self._observe(value)
        if not bool(self.fake_quant_enabled):
            return value
        quantized = self.quantize(value).dequantized
        # Identity input gradient, exact compiler-math forward.
        return _exact_forward_ste(quantized, value)

    def extra_repr(self) -> str:
        return (
            f"base={float(self.base):.9g}, group_size={self.group_size}, "
            f"safety_margin={self.safety_margin}, "
            f"observer_enabled={bool(self.observer_enabled)}"
        )


class P64ToStaticFakeQuant(nn.Module):
    """STE for the strict-integer q+p64 -> fixed-grid INT8 readout.

    This is intentionally separate from :class:`DynamicP64FakeQuant`: the
    recurrent stream keeps its runtime p64 carrier, while an ordinary static
    INT8 consumer receives only this readout copy.
    """

    def __init__(
        self,
        *,
        output_scale: float = 1.0 / 127.0,
        safety_margin: float = 1.0,
        observe: bool = True,
    ) -> None:
        super().__init__()
        self.safety_margin = _require_positive_finite(
            "safety_margin", safety_margin)
        self.register_buffer(
            "output_scale",
            torch.tensor(
                _require_positive_finite("output_scale", output_scale),
                dtype=torch.float64))
        self.register_buffer("observed_absmax", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer(
            "observer_enabled", torch.tensor(bool(observe), dtype=torch.bool),
            persistent=False)
        self.register_buffer(
            "fake_quant_enabled", torch.tensor(True, dtype=torch.bool),
            persistent=False)

    @torch.no_grad()
    def _observe(self, value: Tensor) -> None:
        finite = value.detach().to(torch.float32).to(torch.float64)
        if not bool(torch.all(torch.isfinite(finite))):
            raise ValueError("p64-to-static observer received NaN or infinity")
        current = torch.amax(torch.abs(finite))
        self.observed_absmax.copy_(torch.maximum(self.observed_absmax, current))
        effective = torch.clamp(
            self.observed_absmax * self.safety_margin,
            min=torch.finfo(torch.float64).tiny)
        self.output_scale.copy_(effective / 127.0)

    @torch.no_grad()
    def freeze_scale(self) -> P64ToStaticFakeQuant:
        self.observer_enabled.fill_(False)
        return self

    @torch.no_grad()
    def enable_fake_quant(self, enabled: bool = True) -> P64ToStaticFakeQuant:
        self.fake_quant_enabled.fill_(bool(enabled))
        return self

    def quantize(
        self,
        value: DynamicP64Result,
        *,
        input_base: Tensor | float,
    ) -> StaticInt8Result:
        fixed = P64ToStaticFixedPoint.derive(
            float(torch.as_tensor(input_base)), float(self.output_scale))
        return compiler_exact_p64_to_static(value, fixed)

    def forward(
        self,
        value: Tensor,
        pair: DynamicP64Result,
        *,
        input_base: Tensor | float,
    ) -> Tensor:
        if bool(self.observer_enabled):
            self._observe(value)
        if not bool(self.fake_quant_enabled):
            return value
        exact = self.quantize(pair, input_base=input_base).dequantized
        return _exact_forward_ste(exact, value)


class _DepthARTReadoutMarker(nn.Module):
    """ONNX-stable, mathematically exact scalar-one readout marker."""

    def __init__(self) -> None:
        super().__init__()
        # A registered scalar gives the ONNX initializer a full module path,
        # which lets the profile generator bind every unrolled occurrence to
        # its exact stack/block without relying on graph order.
        self.register_buffer("one", torch.ones((1, 1, 1), dtype=torch.float32))

    def forward(self, value: Tensor) -> Tensor:
        # Shape [1,1,1] is scalar by element count while remaining a legal
        # feature-map constant in N2A.  Amy Chen's compiler path deliberately
        # keeps such scalar constants as immediates rather than vector loads.
        if value.dtype != self.one.dtype or value.device != self.one.device:
            # DepthART's reference recurrence is float32 by construction. Keep
            # the fail-closed check explicit so export never inserts a hidden
            # Cast that obscures the marker initializer's module identity.
            raise RuntimeError(
                "DepthART readout marker requires float32 state on its module device")
        return value * self.one


class _DepthARTTreeComposeMarker(nn.Module):
    """Scalar-one source marker for every associative q+p64 composition."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("one", torch.ones((1, 1, 1), dtype=torch.float32))

    def forward(self, value: Tensor) -> Tensor:
        if value.dtype != self.one.dtype or value.device != self.one.device:
            raise RuntimeError(
                "DepthART tree marker requires float32 state on its module device")
        return value * self.one


class DepthARTDynamicP64Step(nn.Module):
    """QAT scaffold for one ``state = transition * state + injection`` step.

    The forward value is the compiler's integer C128 product and residual-add
    program.  Backpropagation uses the derivative of the mathematically
    equivalent floating recurrence, avoiding the zero gradient of integer
    rounding without pretending the deployed forward is ordinary affine QDQ.

    One instance represents one source-proven stack/timestep/block.  Wider
    selective-scan states must be split by the model into C128 blocks before
    calling this module, exactly like the MLA lowering.
    """

    def __init__(
        self,
        *,
        state_base: float = 1.0 / INT16_LIMIT,
        transition_base: float = 1.0 / INT16_LIMIT,
        injection_base: float = 1.0 / INT16_LIMIT,
        product_base: float = 1.0 / INT16_LIMIT,
        output_base: float = 1.0 / INT16_LIMIT,
        readout_scale: float = 1.0 / 127.0,
        transition_static_scale: float = 1.0 / 127.0,
        transition_static_zero_point: int = 0,
        injection_static_scale: float = 1.0 / 127.0,
        injection_static_zero_point: int = 0,
        safety_margin: float = 1.0,
        observe: bool = True,
        stack_id: str = "",
        timestep: int = 0,
        block_index: int = 0,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        if not stack_id:
            raise ValueError("DepthART dynamic step requires a nonempty stack_id")
        if int(timestep) < 0:
            raise ValueError("DepthART timestep must be non-negative")
        if int(num_blocks) < 1 or not 0 <= int(block_index) < int(num_blocks):
            raise ValueError("invalid DepthART block coordinate")
        self.stack_id = str(stack_id)
        self.timestep = int(timestep)
        self.block_index = int(block_index)
        self.num_blocks = int(num_blocks)
        common = {
            "group_size": 128,
            "safety_margin": safety_margin,
            "observe": observe,
        }
        # Ordinary graph edges arrive at the recurrence on a fixed affine
        # INT8 grid.  Preserve that grid explicitly before converting to the
        # runtime q+p64 representation.
        self.transition_input_quant = StaticAffineInt8FakeQuant(
            scale=transition_static_scale,
            zero_point=transition_static_zero_point,
            safety_margin=safety_margin,
            observe=observe,
        )
        self.injection_input_quant = StaticAffineInt8FakeQuant(
            scale=injection_static_scale,
            zero_point=injection_static_zero_point,
            safety_margin=safety_margin,
            observe=observe,
        )
        self.state_quant = DynamicP64FakeQuant(base=state_base, **common)
        self.transition_quant = DynamicP64FakeQuant(base=transition_base, **common)
        self.injection_quant = DynamicP64FakeQuant(base=injection_base, **common)
        # These modules own calibration/frozen bases for the two physical
        # output boundaries; integer product/add below perform the actual
        # narrowing, rather than applying a second generic fake quantizer.
        self.product_quant = DynamicP64FakeQuant(base=product_base, **common)
        self.output_quant = DynamicP64FakeQuant(base=output_base, **common)
        self.readout_quant = P64ToStaticFakeQuant(
            output_scale=readout_scale,
            safety_margin=safety_margin,
            observe=observe,
        )
        self.tree_state_quant = StaticAffineInt8FakeQuant(
            scale=readout_scale,
            zero_point=0,
            safety_margin=safety_margin,
            observe=observe,
            symmetric=True,
        )
        self.readout_marker = _DepthARTReadoutMarker()
        self.tree_compose_marker = _DepthARTTreeComposeMarker()

    @torch.no_grad()
    def freeze_base(self) -> DepthARTDynamicP64Step:
        # This module is reused for every timestep in one scan block.  The
        # producer's output q+p pair is the next timestep's state input, so
        # those two boundaries must share one static base.  Taking the larger
        # calibrated range is conservative and makes the recurrent ABI exact.
        recurrent_base = torch.maximum(
            self.state_quant.base, self.output_quant.base)
        self.state_quant.base.copy_(recurrent_base)
        self.output_quant.base.copy_(recurrent_base)
        for module in (
            self.state_quant,
            self.transition_quant,
            self.injection_quant,
            self.product_quant,
            self.output_quant,
        ):
            module.freeze_base()
        self.transition_input_quant.freeze_qparams()
        self.injection_input_quant.freeze_qparams()
        # Tree-level static boundaries and the final state readout are one
        # physical grid. Use the larger observed range so both producer paths
        # remain clipping-safe and manifest-identical.
        tree_scale = torch.maximum(
            self.tree_state_quant.scale, self.readout_quant.output_scale)
        self.tree_state_quant.scale.copy_(tree_scale)
        self.tree_state_quant.zero_point.zero_()
        self.readout_quant.output_scale.copy_(tree_scale)
        self.tree_state_quant.freeze_qparams()
        self.readout_quant.freeze_scale()
        return self

    @torch.no_grad()
    def enable_fake_quant(self, enabled: bool = True) -> DepthARTDynamicP64Step:
        for module in (
            self.transition_input_quant,
            self.injection_input_quant,
            self.tree_state_quant,
            self.state_quant,
            self.transition_quant,
            self.injection_quant,
            self.product_quant,
            self.output_quant,
        ):
            module.enable_fake_quant(enabled)
        self.readout_quant.enable_fake_quant(enabled)
        return self

    def _observe_boundaries(
        self, state: Tensor, transition: Tensor, injection: Tensor, *,
        observe_static_inputs: bool = True,
    ) -> None:
        if bool(self.state_quant.observer_enabled):
            self.state_quant._observe(state)
        if bool(self.transition_quant.observer_enabled):
            self.transition_quant._observe(transition)
        if bool(self.injection_quant.observer_enabled):
            self.injection_quant._observe(injection)
        if (observe_static_inputs
                and bool(self.transition_input_quant.observer_enabled)):
            self.transition_input_quant._observe(transition)
        if (observe_static_inputs
                and bool(self.injection_input_quant.observer_enabled)):
            self.injection_input_quant._observe(injection)
        product = state * transition
        if bool(self.product_quant.observer_enabled):
            self.product_quant._observe(product)
        if bool(self.output_quant.observer_enabled):
            self.output_quant._observe(product + injection)

    def quantize_step(
        self,
        state: Tensor,
        transition: Tensor,
        injection: Tensor,
        *,
        state_pair: DynamicP64Result | None = None,
        static_inputs_prepared: bool = False,
    ) -> DynamicP64Result:
        """Return the inspectable compiler-exact q+p64 state result."""

        if (state.shape != transition.shape or state.shape != injection.shape
                or state.ndim < 2 or int(state.shape[-1]) != 128):
            raise ValueError(
                "DepthART dynamic step requires matching [...,128] tensors")
        self._observe_boundaries(
            state, transition, injection,
            observe_static_inputs=not static_inputs_prepared)
        lhs = state_pair if state_pair is not None else self.state_quant.quantize(state)
        transition_static = self.transition_input_quant.quantize(transition)
        injection_static = self.injection_input_quant.quantize(injection)
        transition_pair = compiler_exact_static_int8_to_p64(
            transition_static,
            StaticInt8ToP64FixedPoint.derive(
                transition_static.scale,
                float(self.transition_quant.base),
                zero_point=transition_static.zero_point),
        )
        injection_pair = compiler_exact_static_int8_to_p64(
            injection_static,
            StaticInt8ToP64FixedPoint.derive(
                injection_static.scale,
                float(self.injection_quant.base),
                zero_point=injection_static.zero_point),
        )
        product_fixed = P64ProductFixedPoint.derive(
            float(self.state_quant.base),
            float(self.transition_quant.base),
            float(self.product_quant.base))
        product = compiler_exact_block_p64_product(
            lhs, transition_pair, product_fixed)
        add_fixed = P64ResidualAddFixedPoint.derive(
            float(self.product_quant.base),
            float(self.injection_quant.base),
            float(self.output_quant.base))
        return compiler_exact_residual_p64_add(
            product, injection_pair, add_fixed)

    def prepare_static_inputs(
        self, transition: Tensor, injection: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply each fixed Q/DQ once before static Split/Reshape views.

        Per-tensor affine Q/DQ is elementwise, so it commutes exactly with
        DepthART's timestep Split and C128 reshape.  Hoisting it to the full
        chunk prevents ONNX from duplicating thousands of identical Q/DQ
        pairs without changing a code or gradient.
        """

        return (
            self.transition_input_quant(transition),
            self.injection_input_quant(injection),
        )

    def lift_static_injection(self, injection: Tensor) -> Tensor:
        """Lift a prepared fixed-grid injection onto the recurrent p64 base."""

        if injection.ndim < 2 or int(injection.shape[-1]) != 128:
            raise ValueError("DepthART tree injection lift requires [...,128]")
        # Every tree level is deliberately bounded by this ordinary static
        # grid. Slice/Concat can therefore remain stock compiler operations;
        # q+p64 exists only inside one annotated Mul/Add/readout triplet.
        return self.tree_state_quant(injection)

    def publish_tree_state(self, value: Tensor) -> Tensor:
        """Reassert the shared static state grid after a structural merge.

        Slice, reshape and concatenate are value-preserving but an ordinary
        PTQ compiler is otherwise free to recalibrate their output.  Applying
        the same Q/DQ grid is exactly idempotent for already-published tree
        states and makes the source-QAT contract explicit in ONNX.
        """
        return self.tree_state_quant(value)

    def publish_tree_transition(self, value: Tensor) -> Tensor:
        """Reassert the fixed transition grid after a structural merge."""
        return self.transition_input_quant(value)

    def compose_tree_affine(
        self,
        left_injection: Tensor,
        right_transition: Tensor,
        right_injection: Tensor,
    ) -> Tensor:
        """Strict-INT8 ``a_right*b_left+b_right`` for an affine prefix level.

        Both injection operands arrive on the shared static tree-state grid;
        ``right_transition`` is on its fixed transition Q/DQ grid.  The
        product and add use q+p64 internally, then the marker publishes the
        result back to that static grid.  Consequently no structural operator
        has to preserve a hidden p64 side-band.
        """

        if (left_injection.shape != right_transition.shape
                or left_injection.shape != right_injection.shape
                or left_injection.ndim < 2
                or int(left_injection.shape[-1]) != 128):
            raise ValueError(
                "DepthART tree composition requires matching [...,128] tensors")
        # Keep operand order aligned with the serial contract: recurrent/tree
        # state is lhs and transition is rhs.  Multiplication is commutative,
        # but the explicit order makes the source profile's per-input QDQ
        # proof unambiguous after ONNX import.
        surrogate = left_injection * right_transition + right_injection
        if torch.onnx.is_in_onnx_export():
            # Keep ONNX on the supported real-valued shell.  The exact
            # scalar-one scope marks this Add/Mul pair for manifest-bound AFE
            # replacement and is eliminated as a q+p64 identity.
            return self.tree_state_quant(
                # Route through Module.__call__, just like readout(), so ONNX
                # retains the owning step path.  Calling the child marker
                # directly collapses its scope to /tree_compose_marker and
                # cannot distinguish reused markers in different scan blocks.
                self(
                    surrogate, None, None,
                    tree_compose_marker_only=True,
                ))
        if not bool(self.output_quant.fake_quant_enabled):
            if bool(self.product_quant.observer_enabled):
                self.product_quant._observe(right_transition * left_injection)
            if bool(self.output_quant.observer_enabled):
                self.output_quant._observe(surrogate)
            if bool(self.tree_state_quant.observer_enabled):
                self.tree_state_quant._observe(surrogate)
            return surrogate
        left_static = self.tree_state_quant.quantize(left_injection)
        right_static = self.tree_state_quant.quantize(right_injection)
        left_pair = compiler_exact_static_int8_to_p64(
            left_static,
            StaticInt8ToP64FixedPoint.derive(
                left_static.scale, float(self.output_quant.base),
                zero_point=left_static.zero_point),
        )
        transition_static = self.transition_input_quant.quantize(right_transition)
        transition_pair = compiler_exact_static_int8_to_p64(
            transition_static,
            StaticInt8ToP64FixedPoint.derive(
                transition_static.scale, float(self.transition_quant.base),
                zero_point=transition_static.zero_point),
        )
        right_pair = compiler_exact_static_int8_to_p64(
            right_static,
            StaticInt8ToP64FixedPoint.derive(
                right_static.scale, float(self.output_quant.base),
                zero_point=right_static.zero_point),
        )
        product = compiler_exact_block_p64_product(
            left_pair,
            transition_pair,
            P64ProductFixedPoint.derive(
                float(self.output_quant.base),
                float(self.transition_quant.base),
                float(self.product_quant.base)),
        )
        result = compiler_exact_residual_p64_add(
            product,
            right_pair,
            P64ResidualAddFixedPoint.derive(
                float(self.product_quant.base),
                float(self.output_quant.base),
                float(self.output_quant.base)),
        )
        static_result = compiler_exact_p64_to_static(
            result,
            P64ToStaticFixedPoint.derive(
                float(self.output_quant.base),
                float(self.tree_state_quant.scale)),
        )
        return _exact_forward_ste(static_result.dequantized, surrogate)

    def compose_tree_transition(
        self, left_transition: Tensor, right_transition: Tensor
    ) -> Tensor:
        """Multiply affine transitions and requantize to their fixed INT8 grid."""

        if left_transition.shape != right_transition.shape:
            raise ValueError("DepthART tree transition shapes differ")
        return self.transition_input_quant(left_transition * right_transition)

    def forward(
        self,
        state: Tensor,
        transition: Tensor | None = None,
        injection: Tensor | None = None,
        *,
        readout_only: bool = False,
        tree_compose_marker_only: bool = False,
        static_inputs_prepared: bool = False,
    ) -> Tensor:
        if readout_only and tree_compose_marker_only:
            raise ValueError("DepthART marker modes are mutually exclusive")
        if readout_only:
            return self._readout_impl(state)
        if tree_compose_marker_only:
            return self.tree_compose_marker(state)
        if transition is None or injection is None:
            raise ValueError(
                "DepthART recurrence requires transition and injection")
        if torch.onnx.is_in_onnx_export():
            # Export a standard-op topology shell. The manifest-bound AFE
            # annotation replaces these exact real-valued Mul/Add nodes with
            # the compiler-exact q+p64 program; exporting the integer emulator
            # itself would create an unsupported bit-operation graph.
            if not static_inputs_prepared:
                transition, injection = self.prepare_static_inputs(
                    transition, injection)
            return state * transition + injection
        if not bool(self.output_quant.fake_quant_enabled):
            self._observe_boundaries(
                state, transition, injection,
                observe_static_inputs=not static_inputs_prepared)
            return state * transition + injection
        exact = self.quantize_step(
            state, transition, injection,
            static_inputs_prepared=static_inputs_prepared).dequantized
        surrogate = state * transition + injection
        # Exact integer forward; derivative of the equivalent float recurrence.
        return _exact_forward_ste(exact, surrogate)

    def readout(self, state: Tensor) -> Tensor:
        """Convert one recurrent-state copy to the downstream static grid.

        Re-quantization on the tied recurrent base recovers the exact q+p64
        pair produced by :meth:`forward`.  The recurrence itself never passes
        through this boundary; only the copy used by the C contraction does.
        """

        # Route through Module.__call__ so PyTorch/ONNX retains the owning step
        # scope (stack/block) on the scalar-one marker node.
        return self(state, None, None, readout_only=True)

    def _readout_impl(self, state: Tensor) -> Tensor:
        if torch.onnx.is_in_onnx_export():
            return self.readout_marker(state)
        if (state.ndim < 2 or int(state.shape[-1]) != 128):
            raise ValueError("DepthART p64 readout requires a [...,128] tensor")
        if bool(self.readout_quant.observer_enabled):
            self.readout_quant._observe(state)
        if not bool(self.readout_quant.fake_quant_enabled):
            return state
        pair = self.output_quant.quantize(state)
        exact = self.readout_quant.quantize(
            pair, input_base=self.output_quant.base).dequantized
        return _exact_forward_ste(exact, state)

    def compiler_contract(self) -> dict[str, float | int | str]:
        """Frozen values to serialize into AFE product/add annotations."""

        if any(bool(module.observer_enabled) for module in (
            self.transition_input_quant, self.injection_input_quant,
            self.tree_state_quant,
            self.state_quant, self.transition_quant, self.injection_quant,
            self.product_quant, self.output_quant,
        )) or bool(self.readout_quant.observer_enabled):
            raise RuntimeError("freeze_base() before exporting the DepthART contract")
        return {
            "stack_id": self.stack_id,
            "timestep": self.timestep,
            "block_index": self.block_index,
            "num_blocks": self.num_blocks,
            "state_base": float(self.state_quant.base),
            "transition_base": float(self.transition_quant.base),
            "transition_static_scale": float(self.transition_input_quant.scale),
            "transition_static_zero_point": int(
                self.transition_input_quant.zero_point),
            "product_base": float(self.product_quant.base),
            "injection_base": float(self.injection_quant.base),
            "injection_static_scale": float(self.injection_input_quant.scale),
            "injection_static_zero_point": int(
                self.injection_input_quant.zero_point),
            "output_base": float(self.output_quant.base),
            "readout_scale": float(self.readout_quant.output_scale),
            "tree_state_scale": float(self.tree_state_quant.scale),
            "tree_state_zero_point": int(self.tree_state_quant.zero_point),
        }
