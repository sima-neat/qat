import os
import logging
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from mnist_lit import MNIST_Trainer

from sima_qat.misc import find_latest_file_string


class MNIST_Train(MNIST):
    """ Needed to segment MNIST into train/test sets.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.max_train_len = 50000
        if 'max_samples' in kwargs:
            self.max_train_len = kwargs['max_samples']
            del kwargs['max_samples']
        super().__init__(*args, **kwargs)

    @property
    def raw_folder(self) -> str:
        parent_class_name = self.__class__.__base__.__name__
        return os.path.join(self.root, parent_class_name, "raw")

    def __len__(self) -> int:
        return self.max_train_len

class MNIST_Validation(MNIST):
    """ Needed to segment MNIST into train/test sets.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.test_set_start = 50000

    @property
    def raw_folder(self) -> str:
        parent_class_name = self.__class__.__base__.__name__
        return os.path.join(self.root, parent_class_name, "raw")

    def __len__(self) -> int:
        return super().__len__() - self.test_set_start
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index + self.test_set_start)




def _resolve_freeze_epoch(args: Namespace) -> int | None:
    if args.disable_qat or args.freeze_epoch == -1:
        return None
    freeze_epoch = args.freeze_epoch
    if freeze_epoch is None:
        freeze_epoch = args.epochs - 1 if args.epochs > 1 else None
    if freeze_epoch is not None and not 1 <= freeze_epoch < args.epochs:
        raise ValueError(
            "--freeze-epoch must be between 1 and epochs-1, or -1 to disable recovery"
        )
    return freeze_epoch


def run_train(args: Namespace):
    """ Run the training regimen.
    """
    freeze_epoch = _resolve_freeze_epoch(args)
    if args.resume:
        ckpt = find_latest_file_string(root_path='./checkpoints', tag_str='.ckpt')
        classifier = MNIST_Trainer.load_from_checkpoint(
            ckpt,
            freeze_epoch=freeze_epoch,
        )
    else:
        classifier = MNIST_Trainer(
            export_on_end=args.export_on_end,
            use_qat=(not args.disable_qat),
            freeze_epoch=freeze_epoch,
        )

    classifier.to(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset_train = MNIST_Train(download=args.download, root=args.data, transform=transform, max_samples=args.samples_limit)
    dataset_test = MNIST_Validation(download=args.download, root=args.data, transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=args.batch, pin_memory=True, num_workers=1, persistent_workers=True)
    val_loader = DataLoader(dataset_test, shuffle=False, batch_size=args.batch, pin_memory=True, num_workers=1, persistent_workers=True)

    # This is to exercise the load/store code paths in the QAT code.
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename="mnist_classifier_{epoch}",
        every_n_epochs=1,
        save_top_k=-1,
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs, 
        accelerator=args.device, 
        default_root_dir='.',
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=classifier, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    # Do a pass on the validation set after training. This will exercise the Q/DQ version of the graph.
    trainer.validate(
        model=classifier,
        dataloaders=val_loader,
    )
    return


def get_args():
    """Parse command-line arguments for training the MNIST model 
    using Quantization Aware Training (QAT).
    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(description=f"Train MNIST using QAT")
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch size')
    parser.add_argument('-d', '--data', type=str, default=".", help='Dataset location')
    parser.add_argument('--download', action='store_true', help='Download dataset to specified data path')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use')
    parser.add_argument('--samples-limit', type=int, default=50000, help='Limit train samples to size N')
    parser.add_argument('--export-on-end', action='store_true', help='Export ONNX model at training end')
    parser.add_argument('--disable-qat', action='store_true', help='Disable QAT mode')
    parser.add_argument(
        '--freeze-epoch',
        type=int,
        default=None,
        help='Zero-based epoch for locking QAT grids; defaults to the final epoch, -1 disables recovery',
    )
    parser.add_argument('--resume', action='store_true', help='Resume training from most recent ckpt')
    all_args = parser.parse_args()
    return all_args


if __name__ == "__main__":
    # Set the global seed to be able to replicate results.
    L.seed_everything(42)
    run_args = get_args()

    run_train(run_args)
