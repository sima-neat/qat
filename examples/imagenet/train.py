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
import argparse

import torchvision.datasets as datasets
import os
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

from imagenet_lit import ImageNet_Model_Trainer

from sima_qat.misc import find_latest_file_string

from imagenet_dataset import (
    apply_imagenet_target_transform,
    limit_samples_by_class,
    set_dataset_samples,
)


def _require_split_dir(data_path: str, split: str) -> str:
    split_dir = os.path.join(data_path, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"ImageNet split directory not found: {split_dir}. "
            f"Expected a dataset layout like {data_path}/train and {data_path}/val."
        )
    return split_dir


def get_train_dataloader(data_path, batch_size, samples_limit, workers, crop_size):
    """ Function in order to get the train data loader required for training
        The train data must be in the /train folder under the imagenet data path """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dir = _require_split_dir(data_path, "train")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        ),
    )
    apply_imagenet_target_transform(train_dataset)
    set_dataset_samples(train_dataset, limit_samples_by_class(train_dataset.samples, samples_limit))
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers,
        pin_memory=True, 
        persistent_workers=True
    )

    return train_loader
    
def get_val_dataloader(data_path, batch_size, workers, resize_size, crop_size):
    """ Function in order to get the validation data loader required for validation
        The validation data must be in the /val folder under the imagenet data path"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dir = _require_split_dir(data_path, "val")
    val_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose(
                [transforms.Resize(resize_size), transforms.CenterCrop(crop_size), transforms.ToTensor(), normalize]
            ),
        )
    apply_imagenet_target_transform(val_dataset)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True, 
        persistent_workers=True
    )
    return val_loader


def run_train(args: argparse.Namespace):
    """ Run the training regimen.
    """
    _require_split_dir(args.data, "train")
    _require_split_dir(args.data, "val")
    L.seed_everything(42)

    if args.resume:
        ckpt = find_latest_file_string(root_path='./checkpoints', tag_str='.ckpt')
        if not ckpt:
            raise FileNotFoundError("No checkpoint found under ./checkpoints; run training first or omit --resume.")
        classifier = ImageNet_Model_Trainer.load_from_checkpoint(ckpt)
    else:
        classifier = ImageNet_Model_Trainer(model=args.model, export_on_end=args.export_on_end, use_qat=(not args.disable_qat), 
                                            batch_size=args.batch, device_train=args.device)

    classifier.to(args.device)

    #NOTE : For some models like Inception_v3 the resize_size and the crop_size will be different
    train_loader = get_train_dataloader(args.data, args.batch, args.samples_limit, workers=args.workers, crop_size=224)
    val_loader = get_val_dataloader(args.data, args.batch, workers=args.workers, resize_size=256, crop_size=224)
    
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
    parser.add_argument('--workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--export-on-end', action='store_true', help='Export ONNX model at training end')
    parser.add_argument('--disable-qat', action='store_true', help='Disable QAT mode')
    parser.add_argument('--resume', action='store_true', help='Resume training from most recent checkpoint')
    all_args = parser.parse_args()
    return all_args


if __name__ == "__main__":
    run_args = get_args()
    run_train(run_args)
