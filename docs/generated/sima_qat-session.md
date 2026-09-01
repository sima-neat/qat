# `sima_qat.session`

Source: `sima_qat/session.py`

Customer-facing orchestration for the SiMa QAT lifecycle.

The low-level PT2E functions in :mod:`sima_qat.qat_api` remain available for
framework integrations.  This module gives ordinary PyTorch users one
``nn.Module`` that owns preparation, calibration, FP32-shadow regularization,
freezing, validation, and export.

## Public API

### Class: `QATRecipe`

Validated preparation policy used by :func:`prepare`.

Most users should pass ``recipe="auto"``.  Recipe objects and YAML files
are an escape hatch for SiMa support engineers and qualified model flows.
They intentionally contain policy values only; executable callbacks do not
belong in a portable recipe.

### Function: `load_recipe(recipe)`

Load a built-in recipe name or a data-only YAML recipe.

YAML recipes accept exactly the :class:`QATRecipe` fields plus an optional
``schema_version`` value of ``1``.  Unknown fields fail closed so a typo
cannot silently alter the quantization policy.

### Class: `QATReport`

Structural and optional task-quality result from :meth:`QATSession.validate`.

### Class: `QATCalibrationReport`

Calibration sufficiency and target-grid stability evidence.

### Class: `QATBundle`

Paths and content digests emitted by :meth:`QATSession.export`.

### Class: `QATSession`

A normal ``nn.Module`` with an enforced SiMa QAT lifecycle.

The prepared graph is registered as ``model`` and is therefore the only
model included in ``parameters()`` and ``state_dict()``.  The frozen FP32
teacher is intentionally kept out of module registration: it follows
device moves but does not double checkpoints, optimizer state, or DDP
parameter broadcasts.

### Function: `prepare(model, example_inputs, target, device, recipe, shadow_weight)`

Prepare a model with the recommended SiMa strict-INT8 QAT workflow.

``recipe="auto"`` selects the state-space policy for Mamba, selective
scan, SS2D, and TinyVim modules and the ordinary strict-INT8 policy for
other models.  The input model is not mutated.
