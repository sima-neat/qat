"""ResNet50 QAT regression on the FULL CIFAR10 dataset.

Full torchvision ResNet50 (bottleneck blocks + residual adds + batchnorm), retargeted to
CIFAR10's 10 classes, exercising QAT amenity on the residual-add topology at scale. Unlike
the distilled mini-set tests, this trains on the full CIFAR10 (downloaded on demand) for a
couple of epochs -- enough for the finalized INT8 model to clear the accuracy gate, which
verifies the model genuinely learns *through* QAT rather than reaching peak accuracy.

This module is self-contained: it reuses only the tracked `cifar_classifier` harness
(CIFAR10_Trainer / onnxrt_test), and trains on GPU while finalizing + exporting on CPU.
"""
import copy
import os
from argparse import Namespace

import pytest

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50

pytest.importorskip("pytorch_lightning")
import pytorch_lightning as L
from pytorch_lightning.utilities import disable_possible_user_warnings

import cifar_classifier as cc


# Absolute path to the cached CIFAR dataset (next to this test), so the data is found
# regardless of the test's working directory (see tests/conftest.py); downloaded on demand.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _resnet50_args() -> Namespace:
    return Namespace(
        # 1 epoch is enough to clear the 0.15 accuracy gate with ~2x margin (it reaches ~0.3 by
        # the end of epoch 0); the gate verifies the model learns *through* QAT, not peak accuracy.
        epochs=1,
        batch=128,
        data=_DATA_DIR,
        device='cuda',
        disable_qat=False,
        # 5e-4 converges cleanly to ~0.57 top-1; the larger 2e-3 diverges for this
        # ResNet50 + AdamW + QAT setup, so do not raise it.
        lr=5e-4,
        acc=0.15,
        workers=4,
    )


def _build_full_dataloaders(args: Namespace):
    """Full CIFAR10 (50k train / 10k test) with multi-worker loading; downloads on demand."""
    cifar10_normalization = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cifar10_normalization,
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        cifar10_normalization,
    ])
    dataset_train = CIFAR10(args.data, train=True, download=True, transform=train_transforms)
    dataset_test = CIFAR10(args.data, train=False, download=True, transform=test_transforms)

    workers = getattr(args, 'workers', 4)
    train_dl = DataLoader(dataset_train, batch_size=args.batch, shuffle=True,
                          num_workers=workers, persistent_workers=(workers > 0))
    test_dl = DataLoader(dataset_test, batch_size=args.batch, shuffle=False,
                         num_workers=workers, persistent_workers=(workers > 0))
    return train_dl, test_dl


class _GpuCIFAR10Trainer(cc.CIFAR10_Trainer):
    """Trainer that trains on GPU but finalizes/exports on CPU.

    convert_pt2e() requires a single-device module, but the prepared GraphModule keeps some
    lifted constant tensors on CPU even after Lightning moves the parameters to CUDA. Unifying
    on CPU before finalize/export sidesteps that (finalize + ONNX export are cheap on CPU).
    """

    def _finalize_qat_model(self) -> None:
        self.classifier_model = self.classifier_model.to('cpu')
        super()._finalize_qat_model()

    def to_onnx(self, file_path, input_sample=None, **kwargs) -> None:
        self.train(False)
        print(f"Writing onnx file output to: {file_path}")
        cc.sima_export_onnx(qat_model=self.classifier_model, inputs=self.dummy_inputs,
                            output_file=file_path, device='cpu')


@torch.no_grad()
def _cpu_eval(model, test_dl) -> float:
    """Top-1 accuracy of the finalized (CPU) PyTorch model on the test set."""
    model.eval()
    correct = total = 0
    for x, gt in test_dl:
        logits = model(x)
        correct += int((logits.argmax(dim=-1) == gt).sum())
        total += int(gt.numel())
    return correct / total


def _make_pl_trainer(args: Namespace) -> L.Trainer:
    accelerator = 'cuda' if str(args.device).startswith('cuda') else 'cpu'
    return L.Trainer(
        enable_checkpointing=False,   # needed to avoid torch.fx pickling issues
        logger=False,
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,                    # single device: keep the QAT export graph intact (no DDP)
        default_root_dir='.',
        enable_progress_bar=True,
        log_every_n_steps=20,
    )


def _train_float_and_export(args: Namespace, model) -> float:
    """Train the float ResNet50 (no QAT fine-tuning) for the same schedule and export it to
    ONNX, as a non-quantized baseline alongside the finalized INT8 model. Returns top-1 acc.

    Uses the plain torch.onnx exporter (not sima_export_onnx, which is for the finalized QAT
    GraphModule). Re-seeds first so the float run sees the same data order as the QAT run.
    """
    L.seed_everything(42)
    trainer_module = cc.CIFAR10_Trainer(
        model_name='ResNet50Float',
        classifier_model=model,
        export_on_end=False,        # exported manually below (float model, not a QAT graph)
        use_qat=False,
        batchsz=args.batch,
        lr=args.lr,
    )
    trainer_module.to(args.device)

    train_dl, test_dl = _build_full_dataloaders(args)
    _make_pl_trainer(args).fit(model=trainer_module, train_dataloaders=train_dl, val_dataloaders=test_dl)

    float_model = trainer_module.classifier_model.to('cpu').eval()
    _, test_dl_cpu = _build_full_dataloaders(args)
    acc = _cpu_eval(float_model, test_dl_cpu)

    onnx_file = trainer_module._get_onnx_name()   # 'ResNet50Float_model.onnx'
    print(f"Writing float (no-QAT) onnx file output to: {onnx_file}")
    torch.onnx.export(
        float_model,
        trainer_module.dummy_inputs[0],
        onnx_file,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
    )
    return acc


def _run_resnet50_qat(args: Namespace) -> bool:
    L.seed_everything(42)
    # onnxrt_test() (CPU onnxruntime) calls cc.build_dataloaders internally for its test set, so
    # point it at the full dataset for the duration of this run, then restore it -- this avoids
    # leaking the full-dataset loader into other tests (e.g. densenet) under pytest-randomly.
    orig_build_dataloaders = cc.build_dataloaders
    cc.build_dataloaders = _build_full_dataloaders
    try:
        model = resnet50(weights=None, num_classes=10)
        # Keep an untouched copy (same init weights) to train float-only and export as a baseline.
        float_model = copy.deepcopy(model)

        trainer_module = _GpuCIFAR10Trainer(
            model_name='ResNet50QAT',
            classifier_model=model,
            export_on_end=True,
            use_qat=(not args.disable_qat),
            batchsz=args.batch,
            lr=args.lr,
        )
        trainer_module.to(args.device)

        train_dl, test_dl = _build_full_dataloaders(args)

        # fit() trains on GPU -> on_train_end finalizes on CPU -> on_fit_end exports ONNX on CPU.
        _make_pl_trainer(args).fit(model=trainer_module, train_dataloaders=train_dl, val_dataloaders=test_dl)

        # Finalized model is on CPU now; evaluate it directly, then check the exported ONNX.
        _, test_dl_cpu = _build_full_dataloaders(args)
        pytorch_acc = _cpu_eval(trainer_module.classifier_model, test_dl_cpu)

        onnx_file = trainer_module._get_onnx_name()
        ort_acc = cc.onnxrt_test(args, onnx_file)
        rc = (pytorch_acc >= args.acc) and (ort_acc >= args.acc)

        # Train the same network float-only (no QAT) and export it as a baseline next to the INT8 model.
        float_acc = _train_float_and_export(args, float_model)

        print(f"\n=== ResNet50 full-CIFAR10 summary ===")
        print(f"  float (no-QAT) acc           : {float_acc:.4f}  (exported ResNet50Float_model.onnx)")
        print(f"  PyTorch (finalized INT8) acc : {pytorch_acc:.4f}  (exported ResNet50QAT_model.onnx)")
        print(f"  ONNX runtime accuracy        : {ort_acc:.4f}")
        print(f"  threshold                    : {args.acc}")
        print(f"=== RESULT: clears acc>={args.acc} -> {rc} ===")
        return rc
    finally:
        cc.build_dataloaders = orig_build_dataloaders


@pytest.mark.nightly
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="ResNet50 full-CIFAR10 QAT run requires a CUDA GPU")
def test_resnet50():
    disable_possible_user_warnings()
    rc = _run_resnet50_qat(_resnet50_args())
    assert rc == True
    return


if __name__ == "__main__":
    # Command-line invocation runs the same full-CIFAR10 config; falls back to CPU (much slower)
    # when no GPU is present.
    args = _resnet50_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    _run_resnet50_qat(args)
