from argparse import Namespace
import argparse

import torchvision.datasets as datasets
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from imagenet_lit import ImageNet_Model_Trainer

from sima_qat.misc import find_latest_file_string

def get_train_dataloader(data_path, batch_size, samples_limit, crop_size):
    """ Function in order to get the train data loader required for training
        The train data must be in the /train folder under the imagenet data path """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dir = os.path.join(data_path, "train")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        ),
    )
    train_dataset.samples = train_dataset.samples[:samples_limit]
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=127,
        pin_memory=True, 
        persistent_workers=True
    )

    return train_loader
    
def get_val_dataloader(data_path, batch_size, workers, resize_size, crop_size):
    """ Function in order to get the validation data loader required for validation
        The validation data must be in the /val folder under the imagenet data path"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dir = os.path.join(data_path, "val")
    val_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose(
                [transforms.Resize(resize_size), transforms.CenterCrop(crop_size), transforms.ToTensor(), normalize]
            ),
        )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True, 
        persistent_workers=True
    )
    return val_loader


def _resolve_freeze_epoch(args: argparse.Namespace) -> int | None:
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


def run_train(args: argparse.Namespace):
    """ Run the training regimen.
    """
    freeze_epoch = _resolve_freeze_epoch(args)
    if args.resume:
        ckpt = find_latest_file_string(root_path='./checkpoints', tag_str='.ckpt')
        classifier = ImageNet_Model_Trainer.load_from_checkpoint(
            ckpt,
            freeze_epoch=freeze_epoch,
        )
    else:
        classifier = ImageNet_Model_Trainer(
            model=args.model,
            export_on_end=args.export_on_end,
            use_qat=(not args.disable_qat),
            batch_size=args.batch,
            device_train=args.device,
            freeze_epoch=freeze_epoch,
        )

    classifier.to(args.device)

    #NOTE : For some models like Inception_v3 the resize_size and the crop_size will be different
    train_loader = get_train_dataloader(args.data, args.batch, args.samples_limit, crop_size=224)
    val_loader = get_val_dataloader(args.data, args.batch, workers=4, resize_size=256, crop_size=224)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename=f"imagenet_{args.model}_classifier_{{epoch}}",
        every_n_epochs=1,
        save_top_k=-1,
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs, 
        accelerator=args.device, 
        devices=[0],
        default_root_dir='.',
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=classifier, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    trainer.validate(
        model=classifier,
        dataloaders=val_loader,
    )
    return


def get_args():
    parser = argparse.ArgumentParser(description="Train ImageNet using QAT")
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=1, help='Batch size')
    parser.add_argument('-d', '--data', type=str, default=".", help='Dataset location')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use')
    parser.add_argument('--model', type=str, default="resnet18", help='Torchvision Imagenet Model to be trained')
    parser.add_argument('--samples-limit', type=int, default=1281167, help='Limit train samples to size N')
    parser.add_argument('--export-on-end', action='store_true', help='Export ONNX model at training end')
    parser.add_argument('--disable-qat', action='store_true', help='Disable QAT mode')
    parser.add_argument(
        '--freeze-epoch',
        type=int,
        default=None,
        help='Zero-based epoch for locking QAT grids; defaults to the final epoch, -1 disables recovery',
    )
    parser.add_argument('--resume', action='store_true', help='Resume training from most recent checkpoint')
    all_args = parser.parse_args()
    return all_args


if __name__ == "__main__":
    L.seed_everything(42)
    run_args = get_args()
    run_train(run_args)
