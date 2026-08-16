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
"""ResNet50 QAT regression on the FULL CIFAR10 dataset.

Full torchvision ResNet50 (bottleneck blocks + residual adds + batchnorm), retargeted to
CIFAR10's 10 classes, exercising QAT amenity on the residual-add topology at scale. Unlike
the distilled mini-set tests, this trains on the full CIFAR10 (downloaded on demand) for a
couple of epochs -- enough for the finalized INT8 model to clear the accuracy gate, which
verifies the model genuinely learns *through* QAT rather than reaching peak accuracy.

This module is self-contained: it reuses only the tracked `cifar_classifier` harness
(CIFAR10Trainer / onnxrt_test), and trains on GPU while finalizing + exporting on CPU.
"""
import copy
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as L
import torch
from pytorch_lightning.utilities import disable_possible_user_warnings
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50

from . import cifar_classifier as cc


def _resnet50_args(
    data_dir: Path,
    allow_data_download: bool,
    output_dir: Path,
) -> Namespace:
    return Namespace(
        # One epoch clears the 0.15 learning-through-QAT accuracy gate.
        epochs=1,
        batch=128,
        data=data_dir,
        allow_data_download=allow_data_download,
        output_dir=output_dir,
        device="cuda",
        disable_qat=False,
        # Raising this destabilizes the ResNet50 + AdamW + QAT configuration.
        lr=5e-4,
        acc=0.15,
        workers=4,
        progress_bar=True,
    )


def _build_full_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    """Build deterministic loaders over the full CIFAR-10 dataset."""
    cifar10_normalization = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization,
        ]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), cifar10_normalization]
    )
    dataset_train = CIFAR10(
        args.data,
        train=True,
        download=args.allow_data_download,
        transform=train_transforms,
    )
    dataset_test = CIFAR10(
        args.data,
        train=False,
        download=args.allow_data_download,
        transform=test_transforms,
    )

    workers = args.workers
    train_dl = DataLoader(
        dataset_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=cc.seed_data_worker,
        generator=cc.seeded_generator(),
    )
    test_dl = DataLoader(
        dataset_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=cc.seed_data_worker,
        generator=cc.seeded_generator(),
    )
    return train_dl, test_dl


class _GpuCIFAR10Trainer(cc.CIFAR10Trainer):
    """Train on GPU, then finalize and export the QAT model on CPU."""

    def _finalize_qat_model(self) -> None:
        self.classifier_model = self.classifier_model.to("cpu")
        super()._finalize_qat_model()

    def to_onnx(
        self,
        file_path: str | Path,
        input_sample: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.train(False)
        print(f"Writing ONNX file output to: {file_path}")
        cc.sima_export_onnx(
            qat_model=self.classifier_model,
            inputs=self.dummy_inputs,
            output_file=file_path,
            device="cpu",
        )


@torch.no_grad()
def _cpu_eval(model: torch.nn.Module, test_dl: DataLoader) -> float:
    """Return sample-weighted top-1 accuracy for a finalized CPU model."""
    model.eval()
    correct = 0
    total = 0
    for samples, targets in test_dl:
        logits = model(samples)
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total += int(targets.numel())
    if total == 0:
        raise RuntimeError("The PyTorch validation loader produced no samples.")
    return correct / total


def _make_pl_trainer(args: Namespace) -> L.Trainer:
    accelerator = "cuda" if str(args.device).startswith("cuda") else "cpu"
    return L.Trainer(
        enable_checkpointing=False,
        logger=False,
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        default_root_dir=str(args.output_dir),
        enable_progress_bar=args.progress_bar,
        log_every_n_steps=20,
    )


def _train_float_and_export(
    args: Namespace,
    model: torch.nn.Module,
    loader_builder: cc.LoaderBuilder,
) -> float:
    """Train and export the matching float baseline, returning top-1 accuracy."""
    L.seed_everything(42, workers=True)
    trainer_module = cc.CIFAR10Trainer(
        model_name="ResNet50Float",
        classifier_model=model,
        export_on_end=False,
        use_qat=False,
        batchsz=args.batch,
        output_dir=args.output_dir,
        lr=args.lr,
    )
    trainer_module.to(args.device)

    train_dl, test_dl = loader_builder(args)
    _make_pl_trainer(args).fit(
        model=trainer_module,
        train_dataloaders=train_dl,
        val_dataloaders=test_dl,
    )

    float_model = trainer_module.classifier_model.to("cpu").eval()
    _, test_dl_cpu = loader_builder(args)
    accuracy = _cpu_eval(float_model, test_dl_cpu)

    onnx_path = trainer_module.get_onnx_path()
    print(f"Writing float (no-QAT) ONNX file output to: {onnx_path}")
    torch.onnx.export(
        float_model,
        trainer_module.dummy_inputs[0],
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
    )
    return accuracy


def _run_resnet50_qat(
    args: Namespace,
    loader_builder: cc.LoaderBuilder = _build_full_dataloaders,
) -> bool:
    L.seed_everything(42, workers=True)
    model = resnet50(weights=None, num_classes=10)
    float_model = copy.deepcopy(model)

    trainer_module = _GpuCIFAR10Trainer(
        model_name="ResNet50QAT",
        classifier_model=model,
        export_on_end=True,
        use_qat=not args.disable_qat,
        batchsz=args.batch,
        output_dir=args.output_dir,
        lr=args.lr,
    )
    trainer_module.to(args.device)

    train_dl, test_dl = loader_builder(args)
    _make_pl_trainer(args).fit(
        model=trainer_module,
        train_dataloaders=train_dl,
        val_dataloaders=test_dl,
    )

    _, test_dl_cpu = loader_builder(args)
    pytorch_acc = _cpu_eval(trainer_module.classifier_model, test_dl_cpu)

    onnx_path = trainer_module.get_onnx_path()
    ort_acc = cc.onnxrt_test(args, onnx_path, loader_builder)
    passed = pytorch_acc >= args.acc and ort_acc >= args.acc

    float_acc = _train_float_and_export(args, float_model, loader_builder)

    print("\n=== ResNet50 full-CIFAR10 summary ===")
    print(f"  float (no-QAT) acc           : {float_acc:.4f}")
    print(f"  PyTorch (finalized INT8) acc : {pytorch_acc:.4f}")
    print(f"  ONNX runtime accuracy        : {ort_acc:.4f}")
    print(f"  threshold                    : {args.acc}")
    print(f"=== RESULT: clears acc>={args.acc} -> {passed} ===")
    return passed


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ResNet50 full-CIFAR10 QAT run requires a CUDA GPU",
)
def test_resnet50(
    tmp_path: Path,
    cifar_data_dir: Path,
    allow_data_download: bool,
) -> None:
    with warnings.catch_warnings():
        disable_possible_user_warnings()
        args = _resnet50_args(cifar_data_dir, allow_data_download, tmp_path)
        assert _run_resnet50_qat(args)
