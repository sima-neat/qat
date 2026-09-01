import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sima_qat.depthart import build_depthart_dynamic_p64_compile_profile
from sima_qat.dynamic_p64 import (
    DepthARTDynamicP64Step,
    DynamicP64FakeQuant,
    P64ProductFixedPoint,
    P64ResidualAddFixedPoint,
    P64ToStaticFixedPoint,
    StaticAffineInt8FakeQuant,
    StaticInt8ToP64FixedPoint,
    compiler_exact_block_p64_product,
    compiler_exact_dynamic_p64,
    compiler_exact_p64_to_static,
    compiler_exact_residual_p64_add,
    compiler_exact_static_int8_to_p64,
    decode_p64,
    encode_p64,
)

N2A_COMPILER_ROOT = os.environ.get("N2A_COMPILER_ROOT")
N2A_PERTOKEN = (
    Path(N2A_COMPILER_ROOT) / "pertoken_asym"
    if N2A_COMPILER_ROOT
    else None
)
N2A_AVAILABLE = bool(N2A_PERTOKEN and N2A_PERTOKEN.is_dir())
if N2A_AVAILABLE:
    sys.path.insert(0, str(N2A_PERTOKEN))
    from depthart_dynamic_product import (
        BlockP64ProductFixedPoint,
        block_p64_product,
    )
    from depthart_p64_to_static import (
        P64ToStaticFixedPoint as N2AP64ToStaticFixedPoint,
    )
    from depthart_p64_to_static import p64_to_static
    from depthart_static_int8_to_p64 import (
        StaticInt8ToP64FixedPoint as N2AStaticInt8ToP64FixedPoint,
    )
    from depthart_static_int8_to_p64 import static_int8_to_p64
    from residual_dynamic_add import (
        ResidualAddFixedPoint,
        residual_dynamic_add,
    )
    from runtime_qk_scale_producer import (
        RuntimeQKScaleProducer,
        StaticBaseSpec,
    )

requires_n2a = pytest.mark.skipif(
    not N2A_AVAILABLE,
    reason="set N2A_COMPILER_ROOT to run compiler/QAT integer-parity tests",
)

pytestmark = pytest.mark.regression


def _compiler_oracle(values: np.ndarray, base: float, group_size: int):
    grouped = values.reshape(*values.shape[:-1], values.shape[-1] // group_size, group_size)
    spec = StaticBaseSpec(
        tensor_name="unit",
        base=float(base),
        effective_absmax=float(base) * 32767.0,
        observed_absmax=float(base) * 32767.0,
        safety_margin=1.0,
        source_id="unit",
        source_kind="test-synthetic",
        source_sha256="0" * 64,
        calibration_rows=int(np.prod(grouped.shape[:-1])),
    )
    product = RuntimeQKScaleProducer(spec, allow_test_provenance=True).quantize(grouped)
    return (
        product.codes.reshape(values.shape),
        product.p_int32,
        product.carrier_p64_i8,
        (product.codes.astype(np.float64) * product.dequant_scale).reshape(values.shape),
    )


@pytest.mark.parametrize("group_size", [8, 16, 32, 128])
@requires_n2a
def test_dynamic_p64_matches_n2a_integer_oracle(group_size):
    rng = np.random.default_rng(0xD3A7 + group_size)
    values = rng.normal(0.0, 2.0, (3, 4, 128)).astype(np.float32)
    values[..., ::31] *= 17.0
    base = float(np.max(np.abs(values[:2]))) * 1.1 / 32767.0
    expected_q, expected_p, expected_carrier, expected_dequant = _compiler_oracle(
        values, base, group_size
    )
    actual = compiler_exact_dynamic_p64(
        torch.from_numpy(values), base=base, group_size=group_size
    )
    np.testing.assert_array_equal(actual.codes.numpy(), expected_q)
    np.testing.assert_array_equal(actual.p.numpy(), expected_p)
    np.testing.assert_array_equal(actual.carrier.numpy(), expected_carrier)
    np.testing.assert_array_equal(actual.dequantized.numpy(), expected_dequant.astype(np.float32))


def test_p64_zero_sentinel_round_trip():
    p = torch.tensor([64, 128, 256, 8192, 16384], dtype=torch.int32)
    carrier = encode_p64(p)
    assert int(carrier[-1]) == 0
    torch.testing.assert_close(decode_p64(carrier), p, rtol=0, atol=0)


def test_fake_quant_forward_is_exact_and_gradient_is_identity():
    module = DynamicP64FakeQuant(base=0.01, group_size=8, observe=False)
    value = torch.linspace(-4.0, 5.0, 32).reshape(2, 16).requires_grad_()
    expected = module.quantize(value).dequantized
    actual = module(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)


def test_calibration_base_freezes_but_runtime_p_remains_dynamic():
    module = DynamicP64FakeQuant(group_size=4, safety_margin=1.25)
    module(torch.tensor([[1.0, -2.0, 0.5, 0.0]]))
    expected = 2.0 * 1.25 / 32767.0
    assert float(module.base) == pytest.approx(expected)
    module.freeze_base()
    frozen = module.base.clone()
    low = module.quantize(torch.tensor([[0.01, -0.01, 0.0, 0.0]])).p
    high = module.quantize(torch.tensor([[4.0, -4.0, 0.0, 0.0]])).p
    module(torch.tensor([[100.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(module.base, frozen, rtol=0, atol=0)
    assert int(high) > int(low)


def test_group_size_must_divide_width():
    with pytest.raises(ValueError, match="must divide"):
        compiler_exact_dynamic_p64(torch.randn(2, 15), base=0.01, group_size=8)


@requires_n2a
def test_depthart_block_product_matches_n2a_integer_oracle():
    rng = np.random.default_rng(0xB10C)
    lhs_value = rng.normal(0.0, 0.8, (2, 3, 128)).astype(np.float32)
    rhs_value = rng.normal(0.0, 0.6, (2, 3, 128)).astype(np.float32)
    lhs = compiler_exact_dynamic_p64(
        torch.from_numpy(lhs_value), base=1.0e-2, group_size=128)
    rhs = compiler_exact_dynamic_p64(
        torch.from_numpy(rhs_value), base=1.0e-2, group_size=128)
    qat_fixed = P64ProductFixedPoint.derive(1.0e-2, 1.0e-2, 1.0e-4)
    n2a_fixed = BlockP64ProductFixedPoint.derive(1.0e-2, 1.0e-2, 1.0e-4)
    assert (
        qat_fixed.multiplier, qat_fixed.shift, qat_fixed.pre_shift
    ) == (
        n2a_fixed.multiplier, n2a_fixed.shift, n2a_fixed.pre_shift
    )
    actual = compiler_exact_block_p64_product(lhs, rhs, qat_fixed)
    expected = block_p64_product(
        lhs.codes.numpy(),
        lhs.carrier.numpy().reshape(*lhs.codes.shape[:-1], 1),
        rhs.codes.numpy(),
        rhs.carrier.numpy().reshape(*rhs.codes.shape[:-1], 1),
        n2a_fixed,
    )
    np.testing.assert_array_equal(actual.codes.numpy(), expected.codes)
    np.testing.assert_array_equal(actual.p.numpy(), expected.p_int32)
    np.testing.assert_array_equal(actual.carrier.numpy(), expected.carrier_p64_i8)


@pytest.mark.parametrize(
    "input_scale,output_base,zero_point",
    [(0.002, 3.2e-5, -3), (0.0015, 4.0e-5, 5), (0.0025, 1.1e-5, -7)],
)
@requires_n2a
def test_static_affine_int8_to_p64_matches_n2a_integer_oracle(
    input_scale, output_base, zero_point
):
    rng = np.random.default_rng(0x51A7 + zero_point)
    codes = rng.integers(-128, 128, size=(2, 3, 128), dtype=np.int16).astype(np.int8)
    module = StaticAffineInt8FakeQuant(
        scale=input_scale, zero_point=zero_point, observe=False)
    dequantized = (
        (torch.from_numpy(codes).to(torch.float64) - zero_point) * input_scale
    ).to(torch.float32)
    static = module.quantize(dequantized)
    np.testing.assert_array_equal(static.codes.numpy(), codes)
    qat_fixed = StaticInt8ToP64FixedPoint.derive(
        input_scale, output_base, zero_point=zero_point)
    n2a_fixed = N2AStaticInt8ToP64FixedPoint.derive(
        input_scale, output_base, zero_point=zero_point)
    assert (
        qat_fixed.multiplier, qat_fixed.shift, qat_fixed.int32_bound
    ) == (
        n2a_fixed.multiplier, n2a_fixed.shift, n2a_fixed.int32_bound
    )
    actual = compiler_exact_static_int8_to_p64(static, qat_fixed)
    expected = static_int8_to_p64(codes, n2a_fixed)
    np.testing.assert_array_equal(actual.codes.numpy(), expected.codes)
    np.testing.assert_array_equal(actual.p.numpy(), expected.p_int32)
    np.testing.assert_array_equal(actual.carrier.numpy(), expected.carrier_p64_i8)


@requires_n2a
def test_depthart_residual_add_matches_n2a_integer_oracle():
    rng = np.random.default_rng(0xADD)
    lhs_value = rng.normal(0.0, 0.02, (2, 3, 128)).astype(np.float32)
    rhs_value = rng.normal(0.0, 0.02, (2, 3, 128)).astype(np.float32)
    lhs = compiler_exact_dynamic_p64(
        torch.from_numpy(lhs_value), base=1.0e-4, group_size=128)
    rhs = compiler_exact_dynamic_p64(
        torch.from_numpy(rhs_value), base=1.0e-4, group_size=128)
    qat_fixed = P64ResidualAddFixedPoint.derive(1.0e-4, 1.0e-4, 1.0e-4)
    n2a_fixed = ResidualAddFixedPoint.derive(1.0e-4, 1.0e-4, 1.0e-4)
    assert (
        qat_fixed.lhs_shift, qat_fixed.rhs_shift,
        qat_fixed.lhs_multiplier, qat_fixed.rhs_multiplier,
    ) == (
        n2a_fixed.lhs_shift, n2a_fixed.rhs_shift,
        n2a_fixed.lhs_multiplier, n2a_fixed.rhs_multiplier,
    )
    actual = compiler_exact_residual_p64_add(lhs, rhs, qat_fixed)
    expected = residual_dynamic_add(
        lhs.codes.numpy(),
        lhs.carrier.numpy().reshape(*lhs.codes.shape[:-1], 1),
        rhs.codes.numpy(),
        rhs.carrier.numpy().reshape(*rhs.codes.shape[:-1], 1),
        n2a_fixed,
    )
    np.testing.assert_array_equal(actual.codes.numpy(), expected.codes)
    np.testing.assert_array_equal(actual.p.numpy(), expected.p_int32)
    np.testing.assert_array_equal(actual.carrier.numpy(), expected.carrier_p64_i8)


@requires_n2a
def test_depthart_p64_readout_matches_n2a_integer_oracle():
    rng = np.random.default_rng(0x5A71C)
    values = rng.normal(0.0, 0.08, (2, 3, 128)).astype(np.float32)
    pair = compiler_exact_dynamic_p64(
        torch.from_numpy(values), base=2.5e-5, group_size=128)
    qat_fixed = P64ToStaticFixedPoint.derive(2.5e-5, 7.5e-4)
    n2a_fixed = N2AP64ToStaticFixedPoint.derive(2.5e-5, 7.5e-4)
    assert (
        qat_fixed.multiplier, qat_fixed.shift, qat_fixed.int32_bound
    ) == (
        n2a_fixed.multiplier, n2a_fixed.shift, n2a_fixed.int32_bound
    )
    actual = compiler_exact_p64_to_static(pair, qat_fixed)
    expected = p64_to_static(
        pair.codes.numpy(),
        pair.carrier.numpy().reshape(*pair.codes.shape[:-1], 1),
        n2a_fixed)
    np.testing.assert_array_equal(actual.codes.numpy(), expected)


def test_depthart_step_has_exact_forward_and_equivalent_recurrence_gradient():
    torch.manual_seed(7)
    module = DepthARTDynamicP64Step(
        state_base=1.0e-2,
        transition_base=1.0e-2,
        injection_base=1.0e-4,
        product_base=1.0e-4,
        output_base=1.0e-4,
        observe=False,
        stack_id="decoder.scan0",
        timestep=0,
    )
    state = (torch.randn(2, 128) * 0.7).requires_grad_()
    transition = (torch.randn(2, 128) * 0.5).requires_grad_()
    injection = (torch.randn(2, 128) * 0.02).requires_grad_()
    exact = module.quantize_step(state, transition, injection)
    output = module(state, transition, injection)
    torch.testing.assert_close(output, exact.dequantized, rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(state.grad, transition.detach(), rtol=0, atol=0)
    torch.testing.assert_close(transition.grad, state.detach(), rtol=0, atol=0)
    torch.testing.assert_close(injection.grad, torch.ones_like(injection), rtol=0, atol=0)
    # Re-quantizing the dequantized recurrent value at the same frozen base
    # must preserve the physical q+p pair; the compiler transports it rather
    # than publishing a new static QDQ tensor between timesteps.
    round_trip = module.output_quant.quantize(exact.dequantized)
    torch.testing.assert_close(round_trip.codes, exact.codes, rtol=0, atol=0)
    torch.testing.assert_close(
        round_trip.p.reshape_as(exact.p), exact.p, rtol=0, atol=0)


def test_depthart_static_input_qdq_hoist_is_code_and_gradient_equivalent():
    torch.manual_seed(13)
    module = DepthARTDynamicP64Step(
        state_base=2.0e-4,
        transition_base=3.0e-5,
        injection_base=4.0e-4,
        product_base=2.0e-4,
        output_base=2.0e-4,
        transition_static_scale=4.1e-3,
        transition_static_zero_point=-128,
        injection_static_scale=1.7e-2,
        injection_static_zero_point=-9,
        observe=False,
        stack_id="decoder.scan0",
    )
    state = torch.randn(2, 3, 128) * 0.2
    transition = torch.rand(2, 3, 128)
    injection = torch.randn(2, 3, 128) * 0.7
    direct = module(state, transition, injection)
    transition_q, injection_q = module.prepare_static_inputs(
        transition, injection)
    hoisted = module(
        state, transition_q, injection_q, static_inputs_prepared=True)
    torch.testing.assert_close(hoisted, direct, rtol=0, atol=0)


def test_depthart_readout_is_exact_static_int8_with_identity_gradient():
    torch.manual_seed(9)
    module = DepthARTDynamicP64Step(
        state_base=2.0e-5,
        transition_base=2.0e-5,
        injection_base=2.0e-5,
        product_base=2.0e-5,
        output_base=2.0e-5,
        readout_scale=4.0e-3,
        observe=False,
        stack_id="decoder.scan0",
        timestep=0,
    )
    state = (torch.randn(2, 128) * 0.08).requires_grad_()
    pair = module.output_quant.quantize(state)
    expected = module.readout_quant.quantize(
        pair, input_base=module.output_quant.base).dequantized
    actual = module.readout(state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(state.grad, torch.ones_like(state), rtol=0, atol=0)


def test_depthart_step_freeze_unifies_the_recurrent_input_output_base():
    module = DepthARTDynamicP64Step(
        state_base=1.0e-4,
        transition_base=2.0e-4,
        injection_base=3.0e-4,
        product_base=4.0e-4,
        output_base=5.0e-4,
        observe=False,
        stack_id="decoder.scan0",
        timestep=0,
    )
    module.freeze_base()
    assert float(module.state_quant.base) == float(module.output_quant.base)
    assert module.compiler_contract()["state_base"] == 5.0e-4


def test_depthart_compile_profile_binds_unrolled_onnx_by_marker_dataflow(tmp_path):
    pytest.importorskip("onnx")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = DepthARTDynamicP64Step(
                state_base=2.0e-5,
                transition_base=3.0e-5,
                injection_base=4.0e-5,
                product_base=5.0e-5,
                output_base=2.0e-5,
                readout_scale=3.0e-3,
                transition_static_scale=2.0e-3,
                transition_static_zero_point=-3,
                injection_static_scale=2.5e-3,
                injection_static_zero_point=7,
                observe=False,
                stack_id="decoder.scan0.block0",
            )

        def forward(self, state, transition, injection):
            state1 = self.step(state, transition, injection)
            readout1 = self.step.readout(state1)
            state2 = self.step(state1, transition, injection)
            return readout1 + self.step.readout(state2)

    model = Toy().eval()
    value = torch.randn(1, 2, 3, 128)
    onnx_path = tmp_path / "toy.onnx"
    torch.onnx.export(
        model, (value, value, value), str(onnx_path), opset_version=17)
    import onnx
    exported = onnx.load(str(onnx_path))
    op_types = [node.op_type for node in exported.graph.node]
    assert op_types.count("QuantizeLinear") == 2
    assert op_types.count("DequantizeLinear") == 2
    profile = build_depthart_dynamic_p64_compile_profile(model, onnx_path)
    assert profile["schema"] == "sima-depthart-dynamic-p64-compile-profile/v1"
    assert len(profile["manifest_sha256"]) == 64
    assert [row["timestep"] for row in profile["bindings"]] == [0, 1]
    assert all(row["stack_id"] == "decoder.scan0.block0"
               for row in profile["bindings"])
    source_names = [
        row[role]["source_node"]
        for row in profile["bindings"]
        for role in ("product", "state_add", "readout")
    ]
    assert len(source_names) == len(set(source_names)) == 6
    assert all("readout_marker" in row["readout"]["source_node"]
               for row in profile["bindings"])
    assert all(row["product"]["rhs_input_scale"] == 2.0e-3
               and row["product"]["rhs_input_zero_point"] == -3
               for row in profile["bindings"])
    assert all(row["state_add"]["rhs_input_scale"] == 2.5e-3
               and row["state_add"]["rhs_input_zero_point"] == 7
               for row in profile["bindings"])


def test_depthart_tree_compile_profile_bounds_qp64_by_static_qdq(tmp_path):
    pytest.importorskip("onnx")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = DepthARTDynamicP64Step(
                state_base=2.0e-5,
                transition_base=3.0e-5,
                injection_base=4.0e-5,
                product_base=5.0e-5,
                output_base=2.0e-5,
                readout_scale=3.0e-3,
                transition_static_scale=2.0e-3,
                transition_static_zero_point=-3,
                injection_static_scale=2.5e-3,
                injection_static_zero_point=7,
                observe=False,
                stack_id="decoder.scan0.block0",
            )

        def forward(self, transition, injection):
            transition, injection = self.step.prepare_static_inputs(
                transition, injection)
            state = self.step.lift_static_injection(injection)
            first = self.step.compose_tree_affine(
                state, transition, state)
            return self.step.compose_tree_affine(
                first, transition, state)

    model = Toy().eval()
    value = torch.randn(1, 2, 3, 128)
    onnx_path = tmp_path / "tree_toy.onnx"
    torch.onnx.export(
        model, (value, value), str(onnx_path), opset_version=17)
    profile = build_depthart_dynamic_p64_compile_profile(model, onnx_path)
    assert len(profile["bindings"]) == 2
    assert all(row["scan_primitive"] == "tree_compose"
               for row in profile["bindings"])
    assert all("tree_compose_marker" in row["readout"]["source_node"]
               for row in profile["bindings"])
    assert all(row["product"]["lhs_base"] == 2.0e-5
               and row["product"]["rhs_base"] == 3.0e-5
               and row["product"]["lhs_input_scale"] == 3.0e-3
               and row["product"]["lhs_input_zero_point"] == 0
               and row["product"]["rhs_input_scale"] == 2.0e-3
               and row["product"]["rhs_input_zero_point"] == -3
               for row in profile["bindings"])
    assert all(row["state_add"]["lhs_base"] == 5.0e-5
               and row["state_add"]["rhs_base"] == 2.0e-5
               and row["state_add"]["rhs_input_scale"] == 3.0e-3
               and row["state_add"]["rhs_input_zero_point"] == 0
               for row in profile["bindings"])
    assert all(row["readout"]["input_base"] == 2.0e-5
               and row["readout"]["output_scale"] == 3.0e-3
               for row in profile["bindings"])
    assert all(row["publication"]["scale"] == pytest.approx(3.0e-3)
               and row["publication"]["zero_point"] == 0
               and "QuantizeLinear" in row["publication"]["quantize_source_node"]
               and "DequantizeLinear" in row["publication"]["dequantize_source_node"]
               for row in profile["bindings"])


def test_depthart_tree_profile_rejects_a_provably_zero_state_identity(tmp_path):
    """Do not create a manifest contract for an operator AFE may fold."""

    pytest.importorskip("onnx")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = DepthARTDynamicP64Step(
                observe=False, stack_id="decoder.scan0.block0")

        def forward(self, transition, injection):
            transition, injection = self.step.prepare_static_inputs(
                transition, injection)
            injection = self.step.lift_static_injection(injection)
            initial = torch.zeros_like(injection)
            return self.step.compose_tree_affine(
                initial, transition, injection)

    model = Toy().eval()
    value = torch.randn(3, 1, 2, 128)
    onnx_path = tmp_path / "zero_identity.onnx"
    torch.onnx.export(
        model, (value, value), str(onnx_path), opset_version=17,
        do_constant_folding=False)
    with pytest.raises(RuntimeError, match="state operand is provably zero"):
        build_depthart_dynamic_p64_compile_profile(model, onnx_path)


def test_tree_structural_publication_is_idempotent_and_exported(tmp_path):
    onnx = pytest.importorskip("onnx")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = DepthARTDynamicP64Step(
                readout_scale=3.0e-3,
                transition_static_scale=4.0e-3,
                transition_static_zero_point=-3,
                injection_static_scale=17.0e-3,
                injection_static_zero_point=-9,
                observe=False,
                stack_id="decoder.scan0.block0",
            )

        def forward(self, value):
            state = self.step.lift_static_injection(value)
            merged = torch.cat((state[:, :1], state[:, 1:]), dim=1)
            return self.step.publish_tree_state(merged)

    model = Toy().eval()
    value = torch.randn(1, 4, 1, 128)
    once = model.step.tree_state_quant(value)
    twice = model.step.publish_tree_state(once)
    torch.testing.assert_close(twice, once, rtol=0, atol=0)

    onnx_path = tmp_path / "tree_structural_publication.onnx"
    torch.onnx.export(model, (value,), str(onnx_path), opset_version=17)
    graph = onnx.load(str(onnx_path)).graph
    nodes = list(graph.node)
    concat = next(node for node in nodes if node.op_type == "Concat")
    quantize = next(
        node for node in nodes
        if node.op_type == "QuantizeLinear" and node.input[0] == concat.output[0]
    )
    dequantize = next(
        node for node in nodes
        if node.op_type == "DequantizeLinear"
        and node.input[0] == quantize.output[0]
    )
    assert quantize.input[1:] == dequantize.input[1:]
