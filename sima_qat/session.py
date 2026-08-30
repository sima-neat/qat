# **************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
# **************************************************************************
"""Customer-facing orchestration for the SiMa QAT lifecycle.

The low-level PT2E functions in :mod:`sima_qat.qat_api` remain available for
framework integrations.  This module gives ordinary PyTorch users one
``nn.Module`` that owns preparation, calibration, FP32-shadow regularization,
freezing, validation, and export.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.ao.quantization import disable_fake_quant, enable_fake_quant, enable_observer
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat.qat_api import (
    _MAX_REQUANT_SHIFT,
    _MIN_REQUANT_SHIFT,
    _SHIFT_AWARE_OPS,
    _fake_quant_module,
    _find_output_fake_quant,
    _get_module_device,
    _move_value_to_device,
    _resolve_static_weight,
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from sima_qat.range_refinement import (
    ActivationRangeRefinementReport,
    ActivationRangeSelector,
    refine_activation_ranges,
)

ExampleInputs = Tensor | Sequence[Any]
InputAdapter = Callable[
    [Any], Tensor | Sequence[Any] | tuple[Sequence[Any], Mapping[str, Any]]
]
Evaluator = Callable[[nn.Module, Iterable[Any]], float | Mapping[str, float]]

_WEIGHTED_OPS = _SHIFT_AWARE_OPS | {torch.ops.aten.conv_transpose2d.input}


@dataclass(frozen=True)
class QATRecipe:
    """Validated preparation policy used by :func:`prepare`.

    Most users should pass ``recipe="auto"``.  Recipe objects and YAML files
    are an escape hatch for SiMa support engineers and qualified model flows.
    They intentionally contain policy values only; executable callbacks do not
    belong in a portable recipe.
    """

    name: str
    activation_observer: str = "moving_average"
    full_range_ste: bool = False
    learn_scales: bool = False
    shadow_weight: float = 0.1
    feature_weight: float = 0.1
    activation_ramp_fraction: float = 0.0
    activation_dropout_probability: float = 0.0
    activation_dropout_decay_fraction: float = 1.0
    strict_int8: bool = True
    minimum_calibration_batches: int = 1
    calibration_window_batches: int = 0
    require_shift_tier_stability: bool = False
    maximum_calibration_p99_scale_ratio: float = 1.10
    maximum_calibration_scale_ratio_over_1_25_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.activation_observer not in {"moving_average", "minmax", "histogram"}:
            raise ValueError(
                "activation_observer must be moving_average, minmax, or histogram, "
                f"found {self.activation_observer!r}"
            )
        if self.learn_scales and not self.full_range_ste:
            raise ValueError("learn_scales requires full_range_ste=True")
        if not math.isfinite(self.shadow_weight) or self.shadow_weight < 0:
            raise ValueError("shadow_weight must be a finite non-negative value")
        if not math.isfinite(self.feature_weight) or self.feature_weight < 0:
            raise ValueError("feature_weight must be a finite non-negative value")
        if (
            not math.isfinite(self.activation_ramp_fraction)
            or not 0 <= self.activation_ramp_fraction <= 1
        ):
            raise ValueError("activation_ramp_fraction must be in [0, 1]")
        if (
            not math.isfinite(self.activation_dropout_probability)
            or not 0 <= self.activation_dropout_probability <= 1
        ):
            raise ValueError("activation_dropout_probability must be in [0, 1]")
        if (
            not math.isfinite(self.activation_dropout_decay_fraction)
            or not 0 < self.activation_dropout_decay_fraction <= 1
        ):
            raise ValueError(
                "activation_dropout_decay_fraction must be in (0, 1]"
            )
        if not self.strict_int8:
            raise ValueError(
                "The SiMa customer QAT session currently supports strict INT8 only"
            )
        if self.minimum_calibration_batches < 1:
            raise ValueError("minimum_calibration_batches must be positive")
        if self.calibration_window_batches < 0:
            raise ValueError("calibration_window_batches cannot be negative")
        if (
            self.require_shift_tier_stability
            and self.calibration_window_batches < 1
        ):
            raise ValueError(
                "require_shift_tier_stability needs calibration_window_batches > 0"
            )
        if (
            not math.isfinite(self.maximum_calibration_p99_scale_ratio)
            or self.maximum_calibration_p99_scale_ratio < 1.0
        ):
            raise ValueError(
                "maximum_calibration_p99_scale_ratio must be finite and >= 1"
            )
        if (
            not math.isfinite(
                self.maximum_calibration_scale_ratio_over_1_25_fraction
            )
            or not 0.0
            <= self.maximum_calibration_scale_ratio_over_1_25_fraction
            <= 1.0
        ):
            raise ValueError(
                "maximum_calibration_scale_ratio_over_1_25_fraction must be in [0, 1]"
            )


_BUILTIN_RECIPES: dict[str, QATRecipe] = {
    "strict_int8": QATRecipe(
        name="strict_int8",
        minimum_calibration_batches=64,
        calibration_window_batches=32,
        require_shift_tier_stability=True,
    ),
    "strict_int8_ssm": QATRecipe(
        name="strict_int8_ssm",
        activation_observer="minmax",
        full_range_ste=True,
        learn_scales=False,
        shadow_weight=0.1,
        feature_weight=0.1,
        activation_ramp_fraction=0.25,
        # Recurrent state-space ranges contain rare, legitimate outliers. The
        # qualified DepthART flow needs at least 128 deployment-shaped samples;
        # 64 samples can both clip late SS2D states and cross a power-of-two
        # output-grid tier at the first encoder Conv.
        minimum_calibration_batches=128,
        calibration_window_batches=32,
        require_shift_tier_stability=True,
    ),
}


def load_recipe(recipe: str | Path | QATRecipe) -> QATRecipe:
    """Load a built-in recipe name or a data-only YAML recipe.

    YAML recipes accept exactly the :class:`QATRecipe` fields plus an optional
    ``schema_version`` value of ``1``.  Unknown fields fail closed so a typo
    cannot silently alter the quantization policy.
    """

    if isinstance(recipe, QATRecipe):
        return recipe
    if not isinstance(recipe, (str, Path)):
        raise TypeError(
            f"recipe must be a name, path, or QATRecipe, found {type(recipe)}"
        )

    name = str(recipe)
    if name in _BUILTIN_RECIPES:
        return _BUILTIN_RECIPES[name]
    path = Path(recipe)
    if not path.is_file():
        choices = ", ".join(sorted(_BUILTIN_RECIPES))
        raise ValueError(
            f"Unknown QAT recipe {name!r}; expected {choices} or a YAML file"
        )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"QAT recipe {path} must contain a YAML mapping")
    payload = dict(payload)
    schema_version = payload.pop("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"Unsupported QAT recipe schema_version {schema_version!r}")
    allowed = set(QATRecipe.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown QAT recipe field(s): {', '.join(unknown)}")
    payload.setdefault("name", path.stem)
    return QATRecipe(**payload)


@dataclass(frozen=True)
class QATReport:
    """Structural and optional task-quality result from :meth:`QATSession.validate`."""

    passed: bool
    target: str
    recipe: str
    state: str
    state_space_regions: tuple[str, ...]
    weighted_ops: int
    weighted_ops_covered: int
    activation_fake_quantizers: int
    issues: tuple[str, ...] = ()
    metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def raise_for_failure(self) -> QATReport:
        """Raise a single actionable error when a required validation gate fails."""

        if not self.passed:
            details = "; ".join(self.issues) or "unknown validation failure"
            raise RuntimeError(f"SiMa QAT validation failed: {details}")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return asdict(self)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        regions = len(self.state_space_regions)
        lines = [
            f"SiMa QAT validation: {status}",
            f"Target: {self.target} strict W8A8",
            f"Recipe: {self.recipe}",
            f"Lifecycle state: {self.state}",
            f"Weighted operators covered: {self.weighted_ops_covered}/{self.weighted_ops}",
            f"Activation fake quantizers: {self.activation_fake_quantizers}",
            f"State-space regions detected: {regions}",
        ]
        lines.extend(f"Issue: {issue}" for issue in self.issues)
        return "\n".join(lines)


@dataclass(frozen=True)
class QATCalibrationReport:
    """Calibration sufficiency and target-grid stability evidence."""

    passed: bool
    observed_batches: int
    required_batches: int
    window_batches: int
    windows: tuple[Mapping[str, Any], ...]
    shift_tier_changes_in_last_window: int
    sample_ids: tuple[str, ...]
    ordered_sample_ids_sha256: str | None
    issues: tuple[str, ...] = ()

    def raise_for_failure(self) -> QATCalibrationReport:
        if not self.passed:
            raise RuntimeError(
                "SiMa QAT calibration is not sufficient: "
                + "; ".join(self.issues)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QATBundle:
    """Paths and content digests emitted by :meth:`QATSession.export`."""

    directory: Path
    onnx_path: Path
    manifest_path: Path
    onnx_sha256: str
    quantize_linear_nodes: int
    dequantize_linear_nodes: int

    def __fspath__(self) -> str:
        return str(self.onnx_path)


def _normalize_example_inputs(example_inputs: ExampleInputs) -> tuple[Any, ...]:
    if isinstance(example_inputs, Tensor):
        return (example_inputs,)
    if isinstance(example_inputs, (tuple, list)) and example_inputs:
        return tuple(example_inputs)
    raise ValueError("example_inputs must be a Tensor or a non-empty tuple/list")


_STATE_SPACE_TOKENS = (
    "mamba",
    "selectivescan",
    "selective_scan",
    "statespace",
    "state_space",
    "ss2d",
    "tinyvim",
    "vimblock",
)


def _detect_state_space_regions(model: nn.Module) -> tuple[str, ...]:
    regions: list[str] = []
    for name, module in model.named_modules():
        # Match the module class, not its package path. A package such as
        # ``tinyvim.model`` contains ordinary Conv/BN/ReLU children too; using
        # the path would incorrectly report the entire network as 199
        # state-space regions instead of the root and actual SS2D blocks.
        identity = type(module).__qualname__.lower()
        if any(token in identity for token in _STATE_SPACE_TOKENS):
            regions.append(name or "<root>")
    return tuple(dict.fromkeys(regions))


def _tensor_pairs(quantized: Any, reference: Any) -> list[tuple[Tensor, Tensor]]:
    if isinstance(quantized, Tensor) and isinstance(reference, Tensor):
        if quantized.shape != reference.shape:
            raise RuntimeError(
                "QAT and FP32-shadow outputs have different shapes: "
                f"{tuple(quantized.shape)} != {tuple(reference.shape)}"
            )
        return [(quantized, reference)] if quantized.is_floating_point() else []
    if isinstance(quantized, Mapping) and isinstance(reference, Mapping):
        if quantized.keys() != reference.keys():
            raise RuntimeError(
                "QAT and FP32-shadow outputs have different mapping keys"
            )
        pairs: list[tuple[Tensor, Tensor]] = []
        for key in quantized:
            pairs.extend(_tensor_pairs(quantized[key], reference[key]))
        return pairs
    if isinstance(quantized, (tuple, list)) and isinstance(reference, (tuple, list)):
        if len(quantized) != len(reference):
            raise RuntimeError(
                "QAT and FP32-shadow outputs have different sequence lengths"
            )
        pairs = []
        for quantized_value, reference_value in zip(quantized, reference):
            pairs.extend(_tensor_pairs(quantized_value, reference_value))
        return pairs
    return []


def _as_metric_mapping(value: float | Mapping[str, float]) -> dict[str, float]:
    if isinstance(value, Mapping):
        result = {str(name): float(metric) for name, metric in value.items()}
    else:
        result = {"score": float(value)}
    if not all(math.isfinite(metric) for metric in result.values()):
        raise ValueError("evaluator returned a non-finite metric")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QATSession(nn.Module):
    """A normal ``nn.Module`` with an enforced SiMa QAT lifecycle.

    The prepared graph is registered as ``model`` and is therefore the only
    model included in ``parameters()`` and ``state_dict()``.  The frozen FP32
    teacher is intentionally kept out of module registration: it follows
    device moves but does not double checkpoints, optimizer state, or DDP
    parameter broadcasts.
    """

    _EXPORTABLE_STATES: ClassVar[frozenset[str]] = frozenset(
        {"frozen", "finalized", "exported"}
    )

    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        example_inputs: tuple[Any, ...],
        target: str,
        recipe: QATRecipe,
        state_space_regions: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.model = model
        object.__setattr__(self, "_float_teacher", teacher)
        object.__setattr__(self, "_example_inputs", example_inputs)
        self.target = target
        self.recipe = recipe
        self.state_space_regions = state_space_regions
        self.state = "prepared"
        self.calibration_batches = 0
        self.calibration_report: QATCalibrationReport | None = None
        self._calibration_snapshots: list[dict[str, Any]] = []
        self._calibration_sample_ids: list[str] = []
        self._last_forward: tuple[Any, Any] | None = None
        self._last_loss_terms: dict[str, float] = {}
        self.range_refinement_reports: list[ActivationRangeRefinementReport] = []
        weighted, covered, issues = _weighted_op_coverage(model)
        self._prepared_coverage = (weighted, covered, tuple(issues))
        self._prepared_activation_fake_quantizers = sum(
            isinstance(module, FakeQuantizeBase)
            and module.qscheme
            not in (torch.per_channel_affine, torch.per_channel_symmetric)
            for module in model.modules()
        )

    @property
    def float_teacher(self) -> nn.Module:
        """Frozen FP32 reference used by :meth:`loss` and validation."""

        return object.__getattribute__(self, "_float_teacher")

    @property
    def example_inputs(self) -> tuple[Any, ...]:
        """Example input tuple captured at preparation time."""

        return object.__getattribute__(self, "_example_inputs")

    def _apply(self, fn):
        super()._apply(fn)
        self.float_teacher._apply(fn)
        object.__setattr__(
            self,
            "_example_inputs",
            _apply_to_tensors(self.example_inputs, fn),
        )
        return self

    def train(self, mode: bool = True) -> QATSession:
        if mode and self.state in {"finalized", "exported"}:
            raise RuntimeError("A finalized QAT session is inference-only")
        super().train(mode)
        self.float_teacher.eval()
        return self

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """Restore a training-session checkpoint without serializing the FP32 teacher."""

        model_state = OrderedDict()
        for name, value in state_dict.items():
            if not name.startswith("model."):
                if strict:
                    raise RuntimeError(
                        f"Unexpected QAT session checkpoint key {name!r}"
                    )
                continue
            model_state[name.removeprefix("model.")] = value
        if hasattr(state_dict, "_metadata"):
            model_state._metadata = OrderedDict()
            for name, metadata in state_dict._metadata.items():
                if name == "model":
                    model_state._metadata[""] = metadata
                elif name.startswith("model."):
                    model_state._metadata[name.removeprefix("model.")] = metadata
        result = self.model.load_state_dict(model_state, strict=strict, assign=assign)
        frozen = bool(getattr(self.model, "qat_frozen", torch.tensor([0])).item())
        self.state = "frozen" if frozen else "prepared"
        return result

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        quantized_output = self.model(*args, **kwargs)
        if (
            self.training
            and self.recipe.shadow_weight > 0
            and self.state not in {"finalized", "exported"}
        ):
            with torch.no_grad():
                reference_output = self.float_teacher(*args, **kwargs)
            self._last_forward = (quantized_output, reference_output)
        else:
            self._last_forward = None
        return quantized_output

    def loss(
        self,
        task_loss: Tensor,
        shadow_weight: float | None = None,
        *,
        feature_loss: Tensor | None = None,
        feature_weight: float | None = None,
    ) -> Tensor:
        """Add optional feature and scale-normalized FP32-shadow preservation.

        Call this immediately after the forward that produced ``task_loss``.
        Coefficients default to the selected recipe. ``feature_loss`` is a
        model-defined scalar over matched teacher/student intermediates; the
        lifecycle API deliberately does not prescribe a feature topology.
        """

        if not isinstance(task_loss, Tensor) or task_loss.numel() != 1:
            raise TypeError("task_loss must be a scalar torch.Tensor")
        weight = (
            self.recipe.shadow_weight if shadow_weight is None else float(shadow_weight)
        )
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("shadow_weight must be a finite non-negative value")
        selected_feature_weight = (
            self.recipe.feature_weight
            if feature_weight is None
            else float(feature_weight)
        )
        if not math.isfinite(selected_feature_weight) or selected_feature_weight < 0:
            raise ValueError("feature_weight must be a finite non-negative value")
        if feature_loss is not None and (
            not isinstance(feature_loss, Tensor) or feature_loss.numel() != 1
        ):
            raise TypeError("feature_loss must be a scalar torch.Tensor")
        feature_term = (
            task_loss.new_zeros(()) if feature_loss is None else feature_loss
        )
        if weight == 0:
            self._last_loss_terms = {
                "task": float(task_loss.detach()),
                "feature": float(feature_term.detach()),
                "shadow": 0.0,
            }
            return task_loss + selected_feature_weight * feature_term
        if self._last_forward is None:
            raise RuntimeError(
                "qat.loss() must immediately follow a training-mode qat(...) forward"
            )

        pairs = _tensor_pairs(*self._last_forward)
        if not pairs:
            raise RuntimeError(
                "The model output does not contain matching floating tensors for FP32 preservation"
            )
        consistency_terms = []
        for quantized, reference in pairs:
            reference = reference.to(device=quantized.device, dtype=quantized.dtype)
            normalization = reference.detach().square().mean().clamp_min(1e-12)
            consistency_terms.append(F.mse_loss(quantized, reference) / normalization)
        consistency = torch.stack(consistency_terms).mean()
        self._last_loss_terms = {
            "task": float(task_loss.detach()),
            "feature": float(feature_term.detach()),
            "shadow": float(consistency.detach()),
        }
        self._last_forward = None
        return (
            task_loss
            + selected_feature_weight * feature_term
            + weight * consistency
        )

    def curriculum(
        self,
        step: int,
        total_steps: int,
        ramp_fraction: float | None = None,
        *,
        dropout_probability: float | None = None,
        dropout_decay_fraction: float | None = None,
    ) -> float:
        """Set progressive activation-quantization strength for this update.

        ``step`` is zero based. Weight fake quantization and the frozen target
        grids remain strict throughout. Activation rounding can be blended in
        during the initial ramp and QDrop-style activation bypass can decay to
        zero. The returned strength is useful for logging. ``recipe="auto"``
        uses a 25% ramp for state-space models and no dropout by default.
        """

        if self.state != "frozen":
            raise RuntimeError("qat.curriculum() requires qat.freeze() first")
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(total_steps, int) or total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        fraction = (
            self.recipe.activation_ramp_fraction
            if ramp_fraction is None
            else float(ramp_fraction)
        )
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("ramp_fraction must be in [0, 1]")
        ramp_steps = math.ceil(total_steps * fraction)
        strength = (
            1.0 if ramp_steps == 0 else min(1.0, (step + 1) / ramp_steps)
        )
        initial_dropout = (
            self.recipe.activation_dropout_probability
            if dropout_probability is None
            else float(dropout_probability)
        )
        decay_fraction = (
            self.recipe.activation_dropout_decay_fraction
            if dropout_decay_fraction is None
            else float(dropout_decay_fraction)
        )
        if not math.isfinite(initial_dropout) or not 0 <= initial_dropout <= 1:
            raise ValueError("dropout_probability must be in [0, 1]")
        if not math.isfinite(decay_fraction) or not 0 < decay_fraction <= 1:
            raise ValueError("dropout_decay_fraction must be in (0, 1]")
        decay_steps = max(1, math.ceil(total_steps * decay_fraction))
        dropout = initial_dropout * max(0.0, 1.0 - (step + 1) / decay_steps)
        changed = 0
        for module in self.model.modules():
            if not isinstance(module, FakeQuantizeBase):
                continue
            if module.qscheme in (
                torch.per_channel_affine,
                torch.per_channel_symmetric,
            ):
                continue
            setter = getattr(module, "set_quant_strength", None)
            if setter is not None:
                setter(strength)
                module.set_quantization_dropout_probability(dropout)
                changed += 1
        if fraction and not changed:
            raise RuntimeError(
                "The selected QAT fake quantizer does not support an activation ramp"
            )
        return strength

    def refine_ranges(
        self,
        data: Iterable[Any],
        evaluator: Evaluator,
        *,
        metric: str | None = None,
        factors: Sequence[float] = (1.0, 2.0, 4.0),
        higher_is_better: bool = True,
        min_improvement: float = 0.0,
        policy: str | ActivationRangeSelector = "auto",
        group_by: str = "module",
    ) -> ActivationRangeRefinementReport:
        """Keep a task-improving activation-range candidate or roll it back.

        The default ``policy="auto"`` targets unit-grid Multiply outputs in
        recurrent/state-space regions and excludes grids directly adjacent to
        weighted operators. No graph operators are added or removed. For an
        advanced selection, pass :class:`ActivationRangeSelector`.

        Example::

            report = qat.refine_ranges(
                validation_loader,
                evaluate_depth,
                metric="d1",
            )
        """

        if self.state != "frozen":
            raise RuntimeError("qat.refine_ranges() requires qat.freeze() first")
        report = refine_activation_ranges(
            self.model,
            data,
            evaluator,
            metric=metric,
            factors=factors,
            higher_is_better=higher_is_better,
            min_improvement=min_improvement,
            policy=policy,
            group_by=group_by,
        )
        self.range_refinement_reports.append(report)
        return report

    def calibrate(
        self,
        data: Iterable[Any],
        batches: int | None = None,
        input_adapter: InputAdapter | None = None,
    ) -> QATSession:
        """Collect ranges and verify that target power-of-two tiers are stable.

        When ``batches`` is omitted, the selected recipe's qualified minimum is
        used (128 for automatically detected state-space models). Explicitly
        requesting fewer samples is recorded but cannot pass ``qat.freeze()``.
        Calibration can be extended by calling this method again with more
        representative data.
        """

        if self.state not in {"prepared", "calibrated"}:
            raise RuntimeError(
                "Calibration requires a prepared or calibrated session, "
                f"found state={self.state!r}"
            )
        if batches is None:
            batches = self.recipe.minimum_calibration_batches
        if batches <= 0:
            raise ValueError("batches must be positive")

        was_training = self.training
        self.eval()
        self.model.apply(enable_observer)
        self.model.apply(disable_fake_quant)
        observed = 0
        try:
            device = _get_module_device(self.model)
            with torch.no_grad():
                for batch in data:
                    self._calibration_sample_ids.extend(
                        _sample_ids_from_batch(batch)
                    )
                    args, kwargs = self._calibration_call(batch, input_adapter)
                    args = _move_value_to_device(args, device)
                    kwargs = _move_value_to_device(kwargs, device)
                    self.model(*args, **kwargs)
                    observed += 1
                    total = self.calibration_batches + observed
                    window = self.recipe.calibration_window_batches
                    if window and total % window == 0:
                        self._calibration_snapshots.append(
                            _calibration_snapshot(self.model, total)
                        )
                    if observed >= batches:
                        break
        finally:
            self.model.apply(enable_fake_quant)
            # Deployment calibration owns activation ranges. Batch-1 QAT
            # must not replace representative global extrema with the current
            # training sample. Weight observers remain live so their
            # per-channel grids follow optimizer updates until qat.freeze().
            for module in self.model.modules():
                if not isinstance(module, FakeQuantizeBase):
                    continue
                if module.qscheme in (
                    torch.per_channel_affine,
                    torch.per_channel_symmetric,
                ):
                    enable_observer(module)
                else:
                    # Calibration deliberately ran with fake quantization
                    # disabled, so FakeQuantize.forward did not copy the final
                    # observer qparams into its live buffers. Synchronize them
                    # once here. This is essential for optional LSQ range
                    # tuning: log_scale must start at the calibrated grid, not
                    # at FakeQuantize's construction default of one.
                    scale, zero_point = (
                        module.activation_post_process.calculate_qparams()
                    )
                    scale = scale.to(
                        device=module.scale.device, dtype=module.scale.dtype
                    )
                    zero_point = zero_point.to(
                        device=module.zero_point.device,
                        dtype=module.zero_point.dtype,
                    )
                    module.scale.resize_(scale.shape).copy_(scale)
                    module.zero_point.resize_(zero_point.shape).copy_(zero_point)
                    initialize = getattr(module, "initialize_learned_scale", None)
                    if initialize is not None:
                        initialize()
                    torch.ao.quantization.disable_observer(module)
            self.train(was_training)
        if observed == 0:
            raise RuntimeError("Calibration data produced zero batches")
        self.calibration_batches += observed
        window = self.recipe.calibration_window_batches
        if (
            window
            and (
                not self._calibration_snapshots
                or self._calibration_snapshots[-1]["batch"]
                != self.calibration_batches
            )
        ):
            self._calibration_snapshots.append(
                _calibration_snapshot(self.model, self.calibration_batches)
            )
        self.calibration_report = self._build_calibration_report()
        self.state = "calibrated"
        return self

    def _build_calibration_report(self) -> QATCalibrationReport:
        windows = []
        for previous, current in zip(
            self._calibration_snapshots, self._calibration_snapshots[1:]
        ):
            ratios = []
            for name, scale in previous["activation_scales"].items():
                current_scale = current["activation_scales"].get(name)
                if scale > 0 and current_scale is not None and current_scale > 0:
                    ratios.append(max(current_scale / scale, scale / current_scale))
            previous_tiers = previous["shift_tiers"]
            current_tiers = current["shift_tiers"]
            tier_changes = [
                name
                for name, tier in previous_tiers.items()
                if current_tiers.get(name) != tier
            ]
            ratios.sort()
            p99_index = max(0, math.ceil(0.99 * len(ratios)) - 1)
            windows.append(
                {
                    "from_batch": previous["batch"],
                    "to_batch": current["batch"],
                    "maximum_scale_ratio": max(ratios, default=1.0),
                    "p99_scale_ratio": ratios[p99_index] if ratios else 1.0,
                    "scale_ratio_over_1_25_fraction": (
                        sum(ratio > 1.25 for ratio in ratios) / len(ratios)
                        if ratios else 0.0
                    ),
                    "shift_tier_changes": len(tier_changes),
                    "shift_tier_changed_operators": tier_changes,
                }
            )

        issues = []
        if self.calibration_batches < self.recipe.minimum_calibration_batches:
            issues.append(
                f"observed {self.calibration_batches} calibration batch(es), but "
                f"recipe {self.recipe.name!r} requires at least "
                f"{self.recipe.minimum_calibration_batches}"
            )
        last_tier_changes = windows[-1]["shift_tier_changes"] if windows else 0
        if self.recipe.require_shift_tier_stability:
            if len(self._calibration_snapshots) < 2:
                issues.append(
                    "power-of-two shift stability needs at least two calibration windows"
                )
            elif last_tier_changes:
                issues.append(
                    f"{last_tier_changes} weighted operator shift tier(s) changed "
                    "in the last calibration window"
                )
            elif windows:
                last_window = windows[-1]
                if (
                    last_window["p99_scale_ratio"]
                    > self.recipe.maximum_calibration_p99_scale_ratio
                ):
                    issues.append(
                        "activation-grid p99 scale ratio changed by "
                        f"{last_window['p99_scale_ratio']:.6g} in the last "
                        "calibration window"
                    )
                if (
                    last_window["scale_ratio_over_1_25_fraction"]
                    > self.recipe.maximum_calibration_scale_ratio_over_1_25_fraction
                ):
                    issues.append(
                        "too many activation grids changed by more than 25% in "
                        "the last calibration window: "
                        f"{last_window['scale_ratio_over_1_25_fraction']:.6g}"
                    )

        sample_ids = tuple(self._calibration_sample_ids)
        sample_digest = None
        if sample_ids:
            sample_digest = hashlib.sha256(
                json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return QATCalibrationReport(
            passed=not issues,
            observed_batches=self.calibration_batches,
            required_batches=self.recipe.minimum_calibration_batches,
            window_batches=self.recipe.calibration_window_batches,
            windows=tuple(windows),
            shift_tier_changes_in_last_window=last_tier_changes,
            sample_ids=sample_ids,
            ordered_sample_ids_sha256=sample_digest,
            issues=tuple(issues),
        )

    def _calibration_call(
        self,
        batch: Any,
        input_adapter: InputAdapter | None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        value = input_adapter(batch) if input_adapter is not None else batch
        if (
            input_adapter is not None
            and isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[1], Mapping)
            and isinstance(value[0], (tuple, list))
        ):
            return tuple(value[0]), dict(value[1])

        input_count = len(self.example_inputs)
        if isinstance(value, Tensor):
            return (value,), {}
        if isinstance(value, Mapping):
            for key in ("inputs", "input", "images", "image"):
                if key in value:
                    selected = value[key]
                    return (
                        (selected,) if isinstance(selected, Tensor) else tuple(selected)
                    ), {}
            raise ValueError(
                "Cannot infer model inputs from a mapping batch; provide input_adapter="
            )
        if isinstance(value, (tuple, list)):
            if len(value) < input_count:
                raise ValueError(
                    f"Calibration batch has {len(value)} value(s), but the model has {input_count} input(s)"
                )
            return tuple(value[:input_count]), {}
        raise ValueError(
            f"Cannot infer model inputs from calibration batch type {type(value)}"
        )

    def freeze(self) -> QATSession:
        """Freeze observed activation grids and SiMa-compatible weight shifts."""

        if self.state in self._EXPORTABLE_STATES:
            return self
        if self.state not in {"prepared", "calibrated"}:
            raise RuntimeError(f"Cannot freeze a QAT session in state={self.state!r}")
        if self.calibration_report is None:
            raise RuntimeError(
                "Cannot freeze without qualified calibration. Call qat.calibrate(...) first."
            )
        self.calibration_report.raise_for_failure()
        uninitialized = _uninitialized_activation_observers(self.model)
        if uninitialized:
            raise RuntimeError(
                "Cannot freeze before observers see data. Call qat.calibrate(...) or run "
                f"training forwards first; {len(uninitialized)} activation observer(s) are uninitialized."
            )
        sima_freeze_qat(self.model)
        self.state = "frozen"
        return self

    def finalize(self) -> QATSession:
        """Convert the frozen training graph to inference-only Q/DQ form."""

        if self.state in {"finalized", "exported"}:
            return self
        if self.state != "frozen":
            raise RuntimeError("Finalize requires qat.freeze() first")
        self.model = sima_finalize_qat_model(self.model)
        self.state = "finalized"
        self.eval()
        return self

    def validate(
        self,
        data: Iterable[Any] | None = None,
        evaluator: Evaluator | None = None,
    ) -> QATReport:
        """Validate lifecycle and weighted-op coverage, optionally measuring task quality.

        ``evaluator`` is called as ``evaluator(model, data)`` for both the QAT
        graph and frozen FP32 teacher and must return a float or a mapping of
        metric names to floats.
        """

        if self.state in {"finalized", "exported"}:
            weighted, covered, cached_issues = self._prepared_coverage
            coverage_issues = list(cached_issues)
            activation_fake_quantizers = self._prepared_activation_fake_quantizers
        else:
            weighted, covered, coverage_issues = _weighted_op_coverage(self.model)
            activation_fake_quantizers = sum(
                isinstance(module, FakeQuantizeBase)
                and module.qscheme
                not in (torch.per_channel_affine, torch.per_channel_symmetric)
                for module in self.model.modules()
            )
        issues = list(coverage_issues)
        if self.state not in self._EXPORTABLE_STATES:
            issues.append("QAT grids are not frozen")
        if self.state not in {"finalized", "exported"}:
            non_strict_strengths = [
                float(module.quant_strength)
                for module in self.model.modules()
                if isinstance(module, FakeQuantizeBase)
                and module.qscheme
                not in (torch.per_channel_affine, torch.per_channel_symmetric)
                and hasattr(module, "quant_strength")
                and float(module.quant_strength) != 1.0
            ]
            if non_strict_strengths:
                issues.append(
                    f"{len(non_strict_strengths)} activation fake quantizer(s) "
                    "are below strict strength 1.0"
                )
            nonzero_dropouts = [
                float(module.quantization_dropout_probability)
                for module in self.model.modules()
                if isinstance(module, FakeQuantizeBase)
                and module.qscheme
                not in (torch.per_channel_affine, torch.per_channel_symmetric)
                and hasattr(module, "quantization_dropout_probability")
                and float(module.quantization_dropout_probability) != 0.0
            ]
            if nonzero_dropouts:
                issues.append(
                    f"{len(nonzero_dropouts)} activation fake quantizer(s) "
                    "still have training-only quantization dropout enabled"
                )
        if weighted == 0:
            issues.append("prepared graph contains no supported weighted operators")

        metrics: dict[str, Mapping[str, float]] = {}
        if evaluator is not None:
            if data is None:
                raise ValueError("validate(..., evaluator=...) also requires data")
            was_training = self.training
            self.eval()
            try:
                metrics["qat"] = _as_metric_mapping(evaluator(self.model, data))
                metrics["fp32"] = _as_metric_mapping(
                    evaluator(self.float_teacher, data)
                )
            finally:
                self.train(was_training)

        return QATReport(
            passed=not issues,
            target=self.target,
            recipe=self.recipe.name,
            state=self.state,
            state_space_regions=self.state_space_regions,
            weighted_ops=weighted,
            weighted_ops_covered=covered,
            activation_fake_quantizers=activation_fake_quantizers,
            issues=tuple(issues),
            metrics=metrics,
        )

    def summary(self) -> QATReport:
        """Print and return the current structural validation report."""

        report = self.validate()
        print(report)
        return report

    def explain(self, region: str | None = None) -> str:
        """Explain the automatically selected recipe and state-space policy."""

        selected_regions = self.state_space_regions
        if region is not None:
            selected_regions = tuple(
                name for name in selected_regions if region in name
            )
        lines = [
            f"target={self.target}: strict signed per-tensor INT8 activations and per-channel INT8 weights",
            (
                f"recipe={self.recipe.name}: observer={self.recipe.activation_observer}, "
                f"full_range_ste={self.recipe.full_range_ste}, "
                f"learn_scales={self.recipe.learn_scales}"
            ),
            "weight constraints: Model Compiler power-of-two requantization is locked by qat.freeze()",
            f"fp32 preservation: qat.loss() shadow_weight={self.recipe.shadow_weight}",
            (
                "calibration: at least "
                f"{self.recipe.minimum_calibration_batches} representative batch(es), "
                f"checked every {self.recipe.calibration_window_batches or 'disabled'} batch(es)"
            ),
        ]
        if selected_regions:
            lines.append("state-space regions: " + ", ".join(selected_regions))
        elif region is not None:
            lines.append(f"state-space regions matching {region!r}: none")
        else:
            lines.append("state-space regions: none")
        explanation = "\n".join(lines)
        print(explanation)
        return explanation

    def export(
        self,
        output_directory: str | Path,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        export_device: str | torch.device | None = None,
    ) -> QATBundle:
        """Finalize and export ``model.onnx`` plus a content-bound manifest."""

        if self.state not in self._EXPORTABLE_STATES:
            raise RuntimeError("Export requires qat.freeze() first")
        if self.state == "frozen":
            self.finalize()

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        onnx_path = directory / "model.onnx"
        self.model = sima_export_onnx(
            self.model,
            self.example_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            export_device=export_device,
        )

        import onnx

        onnx_model = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(onnx_model)
        op_counts: dict[str, int] = {}
        for node in onnx_model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        onnx_digest = _sha256(onnx_path)
        report = self.validate()
        manifest = {
            "schema_version": 1,
            "target": self.target,
            "precision": "W8A8",
            "strict_int8_requested": self.recipe.strict_int8,
            "recipe": asdict(self.recipe),
            "calibration": (
                self.calibration_report.to_dict()
                if self.calibration_report is not None else None
            ),
            "range_refinements": [
                refinement.to_dict()
                for refinement in self.range_refinement_reports
            ],
            "example_inputs": [
                _tensor_descriptor(value) for value in self.example_inputs
            ],
            "onnx": {
                "path": onnx_path.name,
                "sha256": onnx_digest,
                "quantize_linear_nodes": op_counts.get("QuantizeLinear", 0),
                "dequantize_linear_nodes": op_counts.get("DequantizeLinear", 0),
            },
            "validation": report.to_dict(),
            "limitations": [
                "This receipt verifies PyTorch preparation and ONNX structure, not Model Compiler or board execution.",
                "Strict full-graph INT8 placement must be audited after stock Model Compiler import.",
            ],
        }
        manifest_path = directory / "qat_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.state = "exported"
        return QATBundle(
            directory=directory,
            onnx_path=onnx_path,
            manifest_path=manifest_path,
            onnx_sha256=onnx_digest,
            quantize_linear_nodes=op_counts.get("QuantizeLinear", 0),
            dequantize_linear_nodes=op_counts.get("DequantizeLinear", 0),
        )


def _apply_to_tensors(value: Any, fn: Callable[[Tensor], Tensor]) -> Any:
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_apply_to_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_apply_to_tensors(item, fn) for item in value]
    if isinstance(value, Mapping):
        return {key: _apply_to_tensors(item, fn) for key, item in value.items()}
    return value


def _sample_ids_from_batch(batch: Any) -> list[str]:
    """Extract ordered, non-executable sample identities when a loader has them."""

    if not isinstance(batch, Mapping):
        return []
    value = None
    for key in ("sample_id", "sample_ids", "id", "ids"):
        if key in batch:
            value = batch[key]
            break
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Tensor):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (tuple, list)):
        return [str(item) for item in value]
    return [str(value)]


def _shift_tier_signature(minimum_ratio: Tensor) -> dict[str, Any]:
    """Digest the exact per-channel compiler right-shift selection."""

    if minimum_ratio.ndim != 1 or minimum_ratio.numel() == 0:
        raise ValueError("minimum_ratio must be a non-empty channel vector")
    if not bool(torch.isfinite(minimum_ratio).all()) or bool(
        (minimum_ratio <= 0).any()
    ):
        raise ValueError("minimum_ratio values must be finite and positive")
    if bool((minimum_ratio > 1.0).any()):
        raise ValueError("minimum_ratio must be retargeted to <= 1 before shifts")
    shifts = torch.floor(-torch.log2(minimum_ratio)).clamp(
        _MIN_REQUANT_SHIFT, _MAX_REQUANT_SHIFT
    ).to(torch.int16)
    shift_bytes = shifts.contiguous().cpu().numpy().tobytes()
    return {
        "minimum_shift": int(shifts.min()),
        "maximum_shift": int(shifts.max()),
        "channel_count": shifts.numel(),
        "channel_shift_sha256": hashlib.sha256(shift_bytes).hexdigest(),
    }


def _calibration_snapshot(model: nn.Module, batch: int) -> dict[str, Any]:
    """Snapshot activation grids and the implied compiler right-shift tiers."""

    activation_scales: dict[str, float] = {}
    name_by_module_id: dict[int, str] = {}
    for name, module in model.named_modules():
        if not isinstance(module, FakeQuantizeBase):
            continue
        if module.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
            continue
        scale, _ = module.activation_post_process.calculate_qparams()
        scalar = float(scale.detach().reshape(-1)[0].cpu())
        if math.isfinite(scalar) and scalar > 0:
            activation_scales[name] = scalar
            name_by_module_id[id(module)] = name

    shift_tiers: dict[str, Mapping[str, Any]] = {}
    if hasattr(model, "graph"):
        contracts = []
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in _SHIFT_AWARE_OPS:
                continue
            input_fq = _fake_quant_module(model, node.args[0])
            weight_fq = _fake_quant_module(model, node.args[1])
            output_fq = _find_output_fake_quant(model, node)
            weight_node = (
                node.args[1].args[0]
                if len(node.args) > 1 and getattr(node.args[1], "args", ())
                else None
            )
            try:
                weight = _resolve_static_weight(model, weight_node)
            except RuntimeError:
                continue
            input_name = name_by_module_id.get(id(input_fq))
            output_name = name_by_module_id.get(id(output_fq))
            if input_name is None or output_name is None or weight_fq is None:
                continue
            input_scale = activation_scales[input_name]
            output_scale = activation_scales[output_name]
            reduce_dims = tuple(range(1, weight.ndim))
            minimum_weight_scale = float(
                weight_fq.activation_post_process.eps.detach().reshape(-1).max().cpu()
            )
            required_scale = torch.clamp(
                weight.detach().abs().amax(dim=reduce_dims).to(torch.float64).cpu()
                / 127.0,
                min=minimum_weight_scale,
            )
            contracts.append(
                {
                    "name": node.name,
                    "input_id": id(input_fq),
                    "output_id": id(output_fq),
                    "input_scale": input_scale,
                    "output_scale": output_scale,
                    "required_scale": required_scale,
                }
            )

        # Mirror sima_freeze_qat's graph-wide output-grid retargeting. A
        # coarsened output can itself feed another weighted operator, so solve
        # the shared activation-domain exponents to a fixed point before
        # deriving any channel shifts.
        grid_exponents: dict[int, int] = {}
        for contract in contracts:
            grid_exponents.setdefault(contract["input_id"], 0)
            grid_exponents.setdefault(contract["output_id"], 0)
        for _ in range(max(1, len(contracts) + 1)):
            changed = False
            for contract in contracts:
                sx = contract["input_scale"] * (
                    2.0 ** grid_exponents[contract["input_id"]]
                )
                sy = contract["output_scale"] * (
                    2.0 ** grid_exponents[contract["output_id"]]
                )
                maximum_ratio = float(
                    ((sx / sy) * contract["required_scale"]).max()
                )
                if maximum_ratio <= 1.0:
                    continue
                exponent = max(1, math.ceil(math.log2(maximum_ratio)))
                grid_exponents[contract["output_id"]] += exponent
                changed = True
            if not changed:
                break
        else:
            raise RuntimeError(
                "Calibration shift-tier solver did not converge"
            )

        for contract in contracts:
            sx = contract["input_scale"] * (
                2.0 ** grid_exponents[contract["input_id"]]
            )
            sy = contract["output_scale"] * (
                2.0 ** grid_exponents[contract["output_id"]]
            )
            minimum_ratio = (sx / sy) * contract["required_scale"]
            shift_tiers[contract["name"]] = {
                "input_grid_exponent": grid_exponents[contract["input_id"]],
                "output_grid_exponent": grid_exponents[contract["output_id"]],
                **_shift_tier_signature(minimum_ratio),
            }
    return {
        "batch": batch,
        "activation_scales": activation_scales,
        "shift_tiers": shift_tiers,
    }


def _tensor_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, Tensor):
        return {"python_type": type(value).__name__}
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
    }


def _uninitialized_activation_observers(model: nn.Module) -> list[str]:
    result = []
    for name, module in model.named_modules():
        if not isinstance(module, FakeQuantizeBase):
            continue
        if module.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
            continue
        observer = module.activation_post_process
        min_val = getattr(observer, "min_val", None)
        max_val = getattr(observer, "max_val", None)
        if min_val is None or max_val is None:
            continue
        if not bool(torch.isfinite(min_val).all() and torch.isfinite(max_val).all()):
            result.append(name)
    return result


def _weighted_op_coverage(model: nn.Module) -> tuple[int, int, list[str]]:
    if not hasattr(model, "graph"):
        # A finalized graph no longer has fake-quant modules to inspect. Its
        # preparation-time coverage was already enforced by freeze().
        return 0, 0, []
    weighted = 0
    covered = 0
    issues = []
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in _WEIGHTED_OPS:
            continue
        weighted += 1
        input_fq = (
            _fake_quant_module(model, node.args[0]) if len(node.args) > 0 else None
        )
        weight_fq = (
            _fake_quant_module(model, node.args[1]) if len(node.args) > 1 else None
        )
        output_fq = _find_output_fake_quant(model, node)
        weight_node = (
            node.args[1].args[0]
            if len(node.args) > 1 and getattr(node.args[1], "args", ())
            else None
        )
        try:
            static_weight = _resolve_static_weight(model, weight_node)
        except RuntimeError:
            static_weight = None
        if (
            input_fq is not None
            and weight_fq is not None
            and output_fq is not None
            and static_weight is not None
            and weight_fq.qscheme
            in (torch.per_channel_affine, torch.per_channel_symmetric)
        ):
            covered += 1
        else:
            issues.append(
                f"weighted operator {node.name!r} is missing input, weight, or output QAT coverage"
            )
    return weighted, covered, issues


def prepare(
    model: nn.Module,
    example_inputs: ExampleInputs,
    target: str = "modalix",
    device: str | torch.device | None = None,
    recipe: str | Path | QATRecipe = "auto",
    shadow_weight: float | None = None,
) -> QATSession:
    """Prepare a model with the recommended SiMa strict-INT8 QAT workflow.

    ``recipe="auto"`` selects the state-space policy for Mamba, selective
    scan, SS2D, and TinyVim modules and the ordinary strict-INT8 policy for
    other models.  The input model is not mutated.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, found {type(model)}")
    normalized_target = str(target).lower()
    if normalized_target not in {"modalix", "sima"}:
        raise ValueError("target must be 'modalix' (or the alias 'sima')")
    normalized_target = "modalix"
    inputs = _normalize_example_inputs(example_inputs)
    state_space_regions = _detect_state_space_regions(model)

    if recipe == "auto":
        selected_recipe = _BUILTIN_RECIPES[
            "strict_int8_ssm" if state_space_regions else "strict_int8"
        ]
    else:
        selected_recipe = load_recipe(recipe)
    if shadow_weight is not None:
        selected_recipe = replace(selected_recipe, shadow_weight=float(shadow_weight))

    try:
        teacher = copy.deepcopy(model)
        training_model = copy.deepcopy(model)
    except Exception as error:
        raise RuntimeError(
            "SiMa QAT preparation could not copy the input model. Remove non-copyable runtime "
            "state or use the low-level sima_prepare_qat_model API."
        ) from error
    teacher.eval().requires_grad_(False)
    selected_device = (
        torch.device(device) if device is not None else _get_module_device(model)
    )
    teacher.to(selected_device)
    prepared = sima_prepare_qat_model(
        training_model,
        inputs,
        selected_device,
        shift_aware=True,
        activation_observer=selected_recipe.activation_observer,
        full_range_ste=selected_recipe.full_range_ste,
        learn_scales=selected_recipe.learn_scales,
    )
    session = QATSession(
        model=prepared,
        teacher=teacher,
        example_inputs=_move_value_to_device(inputs, selected_device),
        target=normalized_target,
        recipe=selected_recipe,
        state_space_regions=state_space_regions,
    )
    session.train()
    return session


__all__ = [
    "ActivationRangeRefinementReport",
    "ActivationRangeSelector",
    "QATBundle",
    "QATCalibrationReport",
    "QATRecipe",
    "QATReport",
    "QATSession",
    "load_recipe",
    "prepare",
]
