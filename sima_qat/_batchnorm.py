"""Preserve BatchNorm literals across PyTorch 2.8 PT2E rewrites.

The upstream QAT patterns hardcode epsilon, and mode rewrites reset momentum.
Keep the captured settings without modifying PyTorch's process-global helpers.
"""

from collections import deque

import torch
from torch import nn
from torch.ao.quantization import FakeQuantizeBase
from torch.fx import Node
from torch.nn.utils.fusion import fuse_conv_bn_weights


_BN = torch.ops.aten.batch_norm.default
_CONVS = (torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default)
_FOLD_META = "sima_batchnorm_fold"
_BN_ID = "sima_batchnorm_id"


def _attributes(node):
    return tuple(
        str(value.target) if isinstance(value, Node) and value.op == "get_attr" else None
        for value in node.args[1:5]
    )


def capture_batchnorm_settings(model, *, record_folding=False):
    settings = {}
    fold_candidates = []
    for node in model.graph.nodes:
        if node.target != _BN:
            continue
        if record_folding:
            node.meta[_BN_ID] = node.name
        key = _attributes(node)
        values = node.args[6:8]
        settings.setdefault(key, []).append((values, node.meta.get(_BN_ID)))
        conv = node.args[0]
        if (
            record_folding
            and isinstance(conv, Node)
            and conv.target in _CONVS
            and isinstance(conv.args[1], Node)
            and conv.args[1].op == "get_attr"
            and all(name is not None for name in key)
        ):
            bias = conv.args[2] if len(conv.args) > 2 else None
            if bias is None or (isinstance(bias, Node) and bias.op == "get_attr"):
                fold_candidates.append(
                    (
                        conv,
                        (
                            str(conv.args[1].target),
                            str(bias.target) if bias is not None else None,
                            key,
                            values[1],
                            node.meta[_BN_ID],
                        ),
                    )
                )
    protected_weights = {
        metadata[0] for _, metadata in fold_candidates if metadata[3] != 1e-5
    }
    for conv, metadata in fold_candidates:
        if metadata[0] in protected_weights:
            conv.meta[_FOLD_META] = metadata
    return settings


def restore_batchnorm_settings(model, settings, *, restore_folding=False):
    if not settings:
        return
    remaining = {key: deque(values) for key, values in settings.items()}
    for node in model.graph.nodes:
        if node.target == _BN and _attributes(node) in remaining:
            (momentum, eps), identity = remaining[_attributes(node)].popleft()
            node.args = (*node.args[:6], momentum, eps, *node.args[8:])
            if identity is not None:
                node.meta[_BN_ID] = identity
        if restore_folding and node.target in _CONVS and _FOLD_META in node.meta:
            _, _, key, eps, _ = node.meta[_FOLD_META]
            # Restrict restoration to this convolution's folded-weight expression,
            # not unrelated user arithmetic that also reads a running variance.
            pending = [node.args[1]]
            visited = set()
            while pending:
                value = pending.pop()
                if not isinstance(value, Node) or value in visited:
                    continue
                visited.add(value)
                if (
                    value.target == torch.ops.aten.add.Tensor
                    and len(value.args) == 2
                    and isinstance(value.args[0], Node)
                    and value.args[0].op == "get_attr"
                    and str(value.args[0].target) == key[3]
                    and isinstance(value.args[1], (int, float))
                    and any(user.target == torch.ops.aten.sqrt.default for user in value.users)
                ):
                    value.args = (value.args[0], eps)
                pending.extend(value.all_input_nodes)
    model.recompile()


def _tensor(model, name):
    if name is None:
        return None
    parent, _, leaf = name.rpartition(".")
    return getattr(model.get_submodule(parent), leaf)


def _replace_tensor(model, name, value):
    parent, _, leaf = name.rpartition(".")
    module = model.get_submodule(parent)
    current = getattr(module, leaf, None)
    if isinstance(current, nn.Parameter):
        value = nn.Parameter(value.detach(), requires_grad=current.requires_grad)
    elif leaf not in module._buffers:
        raise RuntimeError(f"Cannot replace BatchNorm-folded tensor {name!r}")
    setattr(module, leaf, value)


def capture_folded_batchnorm_parameters(model):
    """Compute custom-epsilon folds before PT2E overwrites the original weights."""
    folded = {}
    for node in model.graph.nodes:
        if node.target not in _CONVS or _FOLD_META not in node.meta:
            continue
        weight, bias, key, eps, identity = node.meta[_FOLD_META]
        bn_weight, bn_bias, mean, variance = (_tensor(model, name) for name in key)
        with torch.no_grad():
            folded_weight, folded_bias = fuse_conv_bn_weights(
                _tensor(model, weight), _tensor(model, bias),
                mean, variance, eps, bn_weight, bn_bias,
            )
            fake_quant_node = node.args[1]
            if (
                not isinstance(fake_quant_node, Node)
                or fake_quant_node.op != "call_module"
            ):
                raise RuntimeError(f"Cannot locate weight fake quantizer for {identity!r}")
            fake_quant = model.get_submodule(str(fake_quant_node.target))
            if not isinstance(fake_quant, FakeQuantizeBase):
                raise RuntimeError(f"Cannot locate weight fake quantizer for {identity!r}")
            prepared_weight = fake_quant(folded_weight).detach().clone()
        if identity in folded:
            raise RuntimeError(f"Duplicate BatchNorm fold identity {identity!r}")
        folded[identity] = (weight, prepared_weight, folded_bias)
    return folded


def restore_folded_batchnorm_parameters(model, folded):
    """Correct only the materialized parameters of folds completed by PT2E."""
    remaining = {node.meta.get(_BN_ID) for node in model.graph.nodes if node.target == _BN}
    for node in model.graph.nodes:
        if node.target not in _CONVS or _FOLD_META not in node.meta:
            continue
        weight, _, _, _, identity = node.meta[_FOLD_META]
        original_weight, prepared_weight, folded_bias = folded[identity]
        if original_weight != weight:
            raise RuntimeError(f"BatchNorm fold weight changed for {identity!r}")
        dequantize = node.args[1]
        if not (
            isinstance(dequantize, Node)
            and dequantize.target
            == torch.ops.quantized_decomposed.dequantize_per_channel.default
            and isinstance(dequantize.args[0], Node)
            and dequantize.args[0].op == "get_attr"
        ):
            continue
        qparams = [
            _tensor(model, str(value.target))
            if isinstance(value, Node) and value.op == "get_attr"
            else value
            for value in dequantize.args[1:]
        ]
        quantized_weight = torch.ops.quantized_decomposed.quantize_per_channel.default(
            prepared_weight, *qparams
        )
        _replace_tensor(model, str(dequantize.args[0].target), quantized_weight)
        if identity in remaining:
            continue
        bias = node.args[2]
        if not isinstance(bias, Node) or bias.op != "get_attr":
            raise RuntimeError(f"Cannot locate bias for BatchNorm fold {identity!r}")
        _replace_tensor(model, str(bias.target), folded_bias)
