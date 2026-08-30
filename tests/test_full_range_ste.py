import torch
from torch.ao.quantization.observer import MinMaxObserver

from sima_qat.qat_api import sima_prepare_qat_model
from sima_qat.sima_quantizer import FullRangeSTEFakeQuantize, _LearnedScaleSTE


class TinyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)

    def forward(self, value):
        return self.conv(value)


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


def test_learned_scale_saturation_gradient_is_bounded_by_int8_rail():
    value = torch.tensor([-1000.0, 0.6, 1000.0])
    scale = torch.tensor([2.0], requires_grad=True)
    zero_point = torch.tensor([0.0])
    output = _LearnedScaleSTE.apply(
        value, scale, zero_point, -128, 127, 1.0, 1.0
    )
    output.sum().backward()

    # LSQ: lower rail + in-range rounding error + upper rail.
    expected = -128.0 + (round(0.6 / 2.0) - 0.6 / 2.0) + 127.0
    torch.testing.assert_close(scale.grad, torch.tensor([expected]))


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
