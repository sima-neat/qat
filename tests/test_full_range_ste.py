import torch
from torch.ao.quantization.observer import MinMaxObserver

from sima_qat.qat_api import (
    sima_freeze_qat,
    sima_prepare_qat_model,
    sima_project_qat_to_target_grids,
    sima_thaw_qat_scales,
)
from sima_qat.sima_quantizer import FullRangeSTEFakeQuantize, _LearnedScaleSTE


class TinyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)

    def forward(self, value):
        return self.conv(value)


def test_target_grid_projection_makes_live_qat_equal_to_final_freeze():
    value = torch.randn(1, 3, 4, 4)
    prepared = sima_prepare_qat_model(
        TinyConv(),
        (value,),
        "cpu",
        shift_aware=True,
        full_range_ste=True,
        learn_scales=True,
    )
    prepared(value)
    torch.ao.quantization.disable_observer(prepared)
    live_scales = [
        module
        for module in prepared.modules()
        if hasattr(module, "log_scale") and not module.is_per_channel
    ]
    assert live_scales
    with torch.no_grad():
        live_scales[-1].log_scale.add_(0.2)

    sima_project_qat_to_target_grids(prepared)
    assert not bool(prepared.qat_frozen.item())
    assert all(module.learn_scale for module in live_scales)
    projected = prepared(value)
    projected_qparams = [
        (module.scale.detach().clone(), module.zero_point.detach().clone())
        for module in prepared.modules()
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase)
    ]

    sima_freeze_qat(prepared)
    frozen = prepared(value)
    frozen_qparams = [
        (module.scale.detach().clone(), module.zero_point.detach().clone())
        for module in prepared.modules()
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase)
    ]
    torch.testing.assert_close(frozen, projected, rtol=0, atol=0)
    assert len(frozen_qparams) == len(projected_qparams)
    for projected_pair, frozen_pair in zip(projected_qparams, frozen_qparams):
        torch.testing.assert_close(frozen_pair[0], projected_pair[0], rtol=0, atol=0)
        torch.testing.assert_close(frozen_pair[1], projected_pair[1], rtol=0, atol=0)


def test_frozen_checkpoint_load_and_thaw_use_export_scale_authority():
    value = torch.randn(1, 3, 4, 4)
    prepared = sima_prepare_qat_model(
        TinyConv(),
        (value,),
        "cpu",
        shift_aware=True,
        full_range_ste=True,
        learn_scales=True,
    )
    prepared(value)
    sima_freeze_qat(prepared)
    activation = next(
        module
        for module in prepared.modules()
        if hasattr(module, "log_scale") and not module.is_per_channel
    )

    # Emulate a legacy frozen checkpoint whose retained learned value predates
    # the final power-of-two coarsening.
    state = prepared.state_dict()
    state_key = next(
        name for name, parameter in prepared.named_parameters()
        if parameter is activation.log_scale
    )
    state[state_key] = (activation.scale / 2).log()

    restored = sima_prepare_qat_model(
        TinyConv(),
        (value,),
        "cpu",
        shift_aware=True,
        full_range_ste=True,
        learn_scales=True,
    )
    restored.load_state_dict(state, strict=True)
    restored_activation = dict(restored.named_modules())[
        state_key.removesuffix(".log_scale")
    ]
    torch.testing.assert_close(
        restored_activation.current_learned_scale(),
        restored_activation.scale,
        rtol=0,
        atol=0,
    )
    assert not restored_activation.learn_scale

    sima_thaw_qat_scales(restored, [restored_activation.log_scale])
    torch.testing.assert_close(
        restored_activation.current_learned_scale(),
        restored_activation.scale,
        rtol=0,
        atol=0,
    )
    assert restored_activation.learn_scale
    assert not bool(restored.qat_frozen.item())


def test_prepare_options_override_process_environment(monkeypatch):
    monkeypatch.setenv("SIMA_QAT_ACTIVATION_OBSERVER", "moving_average")
    monkeypatch.setenv("SIMA_QAT_FULL_RANGE_STE", "0")
    monkeypatch.setenv("SIMA_QAT_LEARN_SCALES", "1")
    prepared = sima_prepare_qat_model(
        TinyConv(),
        (torch.randn(1, 3, 4, 4),),
        "cpu",
        activation_observer="minmax",
        full_range_ste=True,
        learn_scales=False,
    )
    activation_fake_quantizers = [
        module
        for module in prepared.modules()
        if isinstance(module, FullRangeSTEFakeQuantize)
        and not module.is_per_channel
    ]
    assert activation_fake_quantizers
    assert all(not module.learn_scale for module in activation_fake_quantizers)
    assert all(
        isinstance(module.activation_post_process, MinMaxObserver)
        for module in activation_fake_quantizers
    )


def test_activation_quantization_strength_curriculum_is_nonpersistent(monkeypatch):
    monkeypatch.setenv("SIMA_QAT_LEARN_SCALES", "0")
    fake_quant = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
    )
    value = torch.tensor([-1.3, -0.1, 0.2, 1.7], dtype=torch.float32)

    # Seed and freeze a deterministic deployment grid.
    fake_quant(value)
    torch.ao.quantization.disable_observer(fake_quant)

    fake_quant.set_quant_strength(0.0)
    torch.testing.assert_close(fake_quant(value), value, rtol=0, atol=0)

    fake_quant.set_quant_strength(1.0)
    strict = fake_quant(value)
    assert not torch.equal(strict, value)

    # Training curriculum state must not leak into the deploy checkpoint. A
    # recreated module always defaults to the strict-INT8 forward.
    assert all("quant_strength" not in key for key in fake_quant.state_dict())
    reloaded = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
    )
    assert reloaded.quant_strength == 1.0


def test_activation_quantization_strength_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("SIMA_QAT_LEARN_SCALES", "0")
    fake_quant = FullRangeSTEFakeQuantize(observer=MinMaxObserver)
    for value in (-0.01, 1.01):
        try:
            fake_quant.set_quant_strength(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid strength {value}")


def test_activation_quantization_dropout_is_training_only_and_nonpersistent():
    fake_quant = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
    )
    value = torch.tensor([-1.3, -0.1, 0.2, 1.7], dtype=torch.float32)
    fake_quant(value)
    torch.ao.quantization.disable_observer(fake_quant)
    strict = fake_quant(value)

    fake_quant.set_quantization_dropout_probability(1.0)
    fake_quant.train()
    torch.testing.assert_close(fake_quant(value), value, rtol=0, atol=0)
    fake_quant.eval()
    torch.testing.assert_close(fake_quant(value), strict, rtol=0, atol=0)
    assert "quantization_dropout_probability" not in fake_quant.state_dict()

    for probability in (-0.01, 1.01):
        try:
            fake_quant.set_quantization_dropout_probability(probability)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid probability {probability}")


def test_activation_quantization_dropout_preserves_identity_input_gradient():
    fake_quant = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
    )
    value = torch.tensor([-2.0, -0.1, 0.2, 3.0], requires_grad=True)
    fake_quant(value.detach())
    torch.ao.quantization.disable_observer(fake_quant)
    fake_quant.set_quantization_dropout_probability(0.5)
    torch.manual_seed(11)
    fake_quant(value).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_learned_scale_gradient_tracks_quantization_strength():
    fake_quant = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        learn_scale=True,
    )
    value = torch.tensor([-1.3, -0.1, 0.2, 1.7], dtype=torch.float32)
    fake_quant(value)
    torch.ao.quantization.disable_observer(fake_quant)

    gradients = []
    for strength in (1.0, 0.25):
        fake_quant.log_scale.grad = None
        fake_quant.set_quant_strength(strength)
        fake_quant(value).sum().backward()
        gradients.append(fake_quant.log_scale.grad.detach().clone())

    torch.testing.assert_close(gradients[1], 0.25 * gradients[0])


def test_relative_scale_learning_preserves_exact_frozen_forward():
    fixed = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        learn_scale=False,
    )
    learned = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        learn_scale=True,
    )
    # 2/255 does not survive an absolute exp(log(scale)) round trip exactly.
    scale = torch.tensor([2.0 / 255.0], dtype=torch.float32)
    zero_point = torch.tensor([-128], dtype=torch.int32)
    for fake_quant in (fixed, learned):
        fake_quant.scale.copy_(scale)
        fake_quant.zero_point.copy_(zero_point)
        torch.ao.quantization.disable_observer(fake_quant)
    learned.enable_relative_scale_learning()
    value = torch.linspace(-0.01, 2.01, 8193)

    assert torch.equal(learned.current_learned_scale(), scale)
    assert torch.equal(learned(value), fixed(value))
    assert "_relative_scale_anchor" not in learned.state_dict()


def test_relative_scale_learning_receives_gradient_and_synchronizes():
    fake_quant = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        learn_scale=True,
    )
    fake_quant.scale.fill_(2.0 / 255.0)
    fake_quant.zero_point.fill_(-128)
    torch.ao.quantization.disable_observer(fake_quant)
    fake_quant.enable_relative_scale_learning()
    value = torch.tensor([0.013, 0.127, 0.991, 1.999])
    fake_quant(value).sum().backward()

    assert fake_quant.log_scale.grad is not None
    assert float(fake_quant.log_scale.grad.abs().max()) > 0
    with torch.no_grad():
        fake_quant.log_scale.add_(0.01)
        expected = fake_quant.current_learned_scale().detach().clone()
        fake_quant.sync_learned_scale()
    torch.testing.assert_close(fake_quant.scale, expected, rtol=0, atol=0)


def test_learned_scale_saturation_gradient_is_bounded_by_int8_rail():
    value = torch.tensor([-1000.0, 0.6, 1000.0])
    scale = torch.tensor([2.0], requires_grad=True)
    zero_point = torch.tensor([0.0])
    output = _LearnedScaleSTE.apply(
        value, scale, zero_point, -128, 127, 1.0, 1.0
    )
    output.sum().backward()

    # LSQ: lower rail + in-range rounding error + upper rail. The custom
    # autograd function deliberately stores the elementwise surrogate in
    # FP16 to halve activation-tape memory, then accumulates it in FP32. Test
    # that exact documented storage contract rather than accidentally asking
    # a half-precision -0.3 to equal its binary32 value.
    stored_terms = torch.tensor(
        [-128.0, round(0.6 / 2.0) - 0.6 / 2.0, 127.0],
        dtype=torch.float16,
    ).float()
    expected = stored_terms.sum().reshape(1)
    torch.testing.assert_close(scale.grad, expected, rtol=0, atol=0)
    # Most importantly, the two saturated outliers contribute bounded rail
    # codes, not the unbounded q-x/s continuation (which would be about zero
    # only through cancellation here and can have the wrong sign in practice).
    assert float(scale.grad.abs()) < 2.0


def test_frozen_learned_scale_uses_memory_bounded_full_range_ste(monkeypatch):
    fake_quant = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        learn_scale=True,
    )
    seed = torch.tensor([-1.3, -0.1, 0.2, 1.7])
    fake_quant(seed)
    torch.ao.quantization.disable_observer(fake_quant)
    fake_quant.log_scale.requires_grad_(False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen scale entered elementwise LSQ path")

    monkeypatch.setattr(_LearnedScaleSTE, "apply", forbidden)
    value = seed.clone().requires_grad_(True)
    fake_quant(value).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_target_code_noise_is_opt_in_and_stays_on_int8_grid(monkeypatch):
    monkeypatch.setenv("SIMA_QAT_LEARN_SCALES", "0")
    fake_quant = FullRangeSTEFakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
    )
    value = torch.tensor([-31.75, -0.5, 0.0, 0.75, 31.75])
    fake_quant.scale.fill_(0.25)
    fake_quant.zero_point.fill_(0)
    torch.ao.quantization.disable_observer(fake_quant)

    reference = fake_quant(value)
    fake_quant.set_target_code_noise_probability(1.0)
    torch.manual_seed(7)
    noisy = fake_quant(value)

    reference_code = torch.round(reference / fake_quant.scale)
    noisy_code = torch.round(noisy / fake_quant.scale)
    assert torch.all((noisy_code - reference_code).abs() <= 1)
    assert torch.allclose(noisy / fake_quant.scale, noisy_code)
    assert int(noisy_code.min()) >= fake_quant.quant_min
    assert int(noisy_code.max()) <= fake_quant.quant_max
    assert "target_code_noise_probability" not in fake_quant.state_dict()
