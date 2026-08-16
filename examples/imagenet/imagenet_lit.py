#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
import copy
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as L
import torch
import torchvision
from torch import nn, optim
from torch.fx.graph_module import GraphModule
from torch.nn import CrossEntropyLoss

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "build" / "examples" / "imagenet"


def _module_device(module: nn.Module) -> torch.device:
    """Return the device that owns a module's state."""
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _resolve_weights(model_name: str, weights: str | None):
    """Resolve a stable CLI/checkpoint weight policy to a torchvision enum."""
    if weights is None or str(weights).lower() == "none":
        return None
    weights_enum = torchvision.models.get_model_weights(model_name)
    if str(weights).upper() == "DEFAULT":
        return weights_enum.DEFAULT
    try:
        return weights_enum[str(weights)]
    except KeyError as error:
        choices = ", ".join(member.name for member in weights_enum)
        raise ValueError(
            f"Unknown weights policy {weights!r} for {model_name}. "
            f"Use DEFAULT, none, or one of: {choices}."
        ) from error


# Some parts adapted from
# https://github.com/MadryLab/pytorch-lightning-imagenet/blob/main/imagenet.py
class ImageNet_Model_Trainer(L.LightningModule):
    def __init__(
        self,
        model: str,
        use_qat: bool = True,
        export_on_end: bool = False,
        batch_size: int = 50,
        device_train: str = "cuda",
        output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
        weights: str | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.exports_dir = self.output_dir / "exports"
        self.graphs_dir = self.output_dir / "graphs"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.imagenet_model = torchvision.models.get_model(
            model,
            weights=_resolve_weights(model, weights),
        )
        self.prev_epoch_step = 0
        self.val_correct = 0
        self.val_samples = 0
        self.use_qat = use_qat
        self._qat_prepared = False
        self.export_on_end = export_on_end
        self.dump_fx_graphs = True
        self.dummy_inputs = (torch.randn(1, 3, 224, 224),)
        self.loss_fn = CrossEntropyLoss()
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.batch_size = batch_size
        self.device_train = device_train
        self.weights_policy = weights
        self.save_hyperparameters()

    def configure_optimizers(self):
        return [
            optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        ]

    def forward(self, imgs):
        return self.imagenet_model(imgs)

    def training_step(self, batch, batch_idx):
        del batch_idx
        images, target = batch
        output = self(images)
        loss_train = self.loss_fn(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(
            "train_loss",
            loss_train,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        del batch_idx
        images, target = batch
        output = self(images)
        loss_value = self.loss_fn(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(f"{prefix}_loss", loss_value, prog_bar=True)
        self.log(f"{prefix}_acc1", acc1, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5)
        if prefix == "val":
            predictions = output.argmax(dim=1)
            self.val_correct += int(predictions.eq(target).sum().item())
            self.val_samples += int(target.numel())
        return loss_value

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def on_validation_end(self) -> None:
        super().on_validation_end()
        if self.val_samples:
            top1_acc = self.val_correct / self.val_samples
            print(f"Validation top-1 accuracy: {top1_acc:.6f}")
        else:
            print("Validation top-1 accuracy unavailable: no samples were evaluated.")
        self.val_correct = 0
        self.val_samples = 0
        self.prev_epoch_step = self.global_step

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Compute top-k percentage accuracy for each requested k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            correct = pred.t().eq(target.view(1, -1).expand(maxk, -1))
            return [
                correct[:k].reshape(-1).float().sum().mul(100.0 / batch_size)
                for k in topk
            ]

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.use_qat and not self._qat_prepared:
            self._prepare_qat()
        elif not self.use_qat:
            print("Tracing float model to an FX graph ...")
            self._dump_fx_graph("compiled_graph.txt")

    def on_train_end(self) -> None:
        super().on_train_end()
        self._finalize_qat_model()

    def on_train_epoch_start(self) -> None:
        self.train(True)

    def _prepare_qat(self) -> None:
        if self._qat_prepared:
            return
        prepared = sima_prepare_qat_model(
            input_graph=self.imagenet_model,
            inputs=self.dummy_inputs,
            device=_module_device(self.imagenet_model),
        )
        self.imagenet_model = prepared
        self._qat_prepared = True
        self._dump_fx_graph("prepare_fx_qat_graph.txt")

    def _finalize_qat_model(self) -> None:
        self.train(False)
        if not self.use_qat:
            return
        if not self._qat_prepared:
            raise RuntimeError("QAT must be prepared before it can be finalized.")
        self.imagenet_model = sima_finalize_qat_model(self.imagenet_model)
        self._dump_fx_graph("final_fx_qat_graph.txt")

    def on_fit_end(self) -> None:
        if self.export_on_end:
            suffix = "qat" if self.use_qat else "float"
            self.to_onnx(file_path=f"{self.model_name}_{suffix}.onnx")

    def _dump_fx_graph(self, fname: str) -> None:
        if not self.dump_fx_graphs:
            return
        if isinstance(self.imagenet_model, GraphModule):
            symbolic_traced = self.imagenet_model
        else:
            from torch.fx import symbolic_trace

            symbolic_traced = symbolic_trace(self.imagenet_model)

        graph_path = self.graphs_dir / fname
        print(f"Generating graph dump to file: {graph_path}")
        with graph_path.open("wt", encoding="utf-8") as graph_file:
            print(symbolic_traced.graph, file=graph_file)

    def to_onnx(
        self,
        file_path: str | Path,
        input_sample: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Export either a finalized QAT graph or a float model to ONNX."""
        self.imagenet_model.train(False)
        output_path = Path(file_path).expanduser()
        if not output_path.is_absolute():
            output_path = self.exports_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing ONNX model to: {output_path}")

        inputs = input_sample if input_sample is not None else self.dummy_inputs
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if self.use_qat:
            self.imagenet_model = sima_export_onnx(
                qat_model=self.imagenet_model,
                inputs=inputs,
                output_file=output_path,
                device=_module_device(self.imagenet_model),
            )
            return

        export_model = copy.deepcopy(self.imagenet_model).cpu().eval()
        cpu_inputs = tuple(
            value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for value in inputs
        )
        export_kwargs = {
            "export_params": True,
            "opset_version": 17,
            "do_constant_folding": True,
            **kwargs,
        }
        with torch.no_grad():
            torch.onnx.export(export_model, cpu_inputs, output_path, **export_kwargs)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Prepare the QAT scaffold before Lightning restores its tensors."""
        if self.use_qat:
            self._prepare_qat()
        return super().on_load_checkpoint(checkpoint)
