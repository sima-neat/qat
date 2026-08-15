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
import os

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import optim, nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from torch.fx.graph_module import GraphModule

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_lightning as L

from sima_qat.qat_api import (sima_prepare_qat_model, 
                              sima_finalize_qat_model, 
                              sima_export_onnx)


# Some parts adapted from https://github.com/MadryLab/pytorch-lightning-imagenet/blob/main/imagenet.py
class ImageNet_Model_Trainer(L.LightningModule):
    # Some settings taken from https://github.com/MadryLab/pytorch-lightning-imagenet/blob/main/imagenet.py
    def __init__(self, model: str, use_qat: bool = True, export_on_end: bool = False, 
                 batch_size: int = 50, device_train: str = 'cuda'):
        super().__init__()
        self.model_name = model
        self.imagenet_model = torchvision.models.__dict__[model](pretrained = True)
        self.prev_epoch_step = 0
        self.val_accuracy = 0
        self.val_batch_count = 0
        self.use_qat = use_qat
        self.export_on_end = export_on_end
        self.dump_fx_graphs = True
        self.dummy_inputs = (torch.randn(1, 3, 224, 224), )
        self.loss_fn = CrossEntropyLoss()
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.batch_size = batch_size
        self.workers = 4
        self.device_train = device_train
        # Call this last once all init has been done
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [optimizer]
    
    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.imagenet_model(imgs)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = self.loss_fn(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self(images)
        loss_val = self.loss_fn(output, target)
        self.log("val_loss", loss_val, prog_bar=True)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.val_accuracy += acc1
        self.val_batch_count += 1
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")
    
    def on_validation_end(self) -> None:
        super().on_validation_end()
        top1_acc = self.val_accuracy / self.val_batch_count
        print(f"Validation top-1 accuracy: {top1_acc}")
        self.val_accuracy = 0
        self.val_batch_count = 0
        self.prev_epoch_step = self.global_step 
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")
   
    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def on_train_start(self) -> None:
        super().on_train_start()
        if self.use_qat:
            self._prepare_qat()
        else:
            # Do a compile so we can see an FX graph
            print(f"Compiling model to FX graph ...")
            self._dump_fx_graph('compiled_graph.txt')
        pass
    
    def on_train_end(self) -> None:
        super().on_train_end()
        self._finalize_qat_model()

    def on_train_epoch_start(self) -> None:
        # For some reason Lightning doesn't switch to train mode hence, we ensure it switches to train mode here
        self.train(True)

    def _prepare_qat(self) -> None:
        m = sima_prepare_qat_model(input_graph=self.imagenet_model, inputs=self.dummy_inputs, device=self.device_train)
        # Now replace our model
        setattr(self, 'imagenet_model', m)
        self._dump_fx_graph('prepare_fx_qat_graph.txt')
        
    def _finalize_qat_model(self) -> None:
        self.train(False)
        # If we are running in QAT mode, we first convert to a quantized graph.
        if self.use_qat:
            m = sima_finalize_qat_model(self.imagenet_model)
            # Now replace our model
            setattr(self, 'imagenet_model', m)
            self._dump_fx_graph('final_fx_qat_graph.txt')
        return

    def on_fit_end(self) -> None:
        if self.export_on_end:
            self.to_onnx(file_path=f"exported_model_{self.model_name}.onnx")
            
    def _dump_fx_graph(self, fname: str) -> None:
        if not self.dump_fx_graphs:
            return
        if not isinstance(self.imagenet_model, GraphModule):
            from torch.fx import symbolic_trace
            # Symbolic tracing frontend - captures the semantics of the module
            symbolic_traced : torch.fx.GraphModule = symbolic_trace(self.imagenet_model)
            # If the graph is not already in FX format, we skip this entire step.
            # return
        else:
            symbolic_traced = self.imagenet_model

        # High-level intermediate representation (IR) - Graph representation
        print(f"Generating graph dump to file: {fname}")
        with open(fname, 'wt') as f:
            print(symbolic_traced.graph, file=f)

    def to_onnx(self, file_path: str | Path, input_sample: Any | None = None, **kwargs: Any) -> None:
        """ This function needs to be overridden in the case of QAT, since export gets tricky
            and specialized.
        """
        self.imagenet_model.train(False)
        self.imagenet_model = sima_export_onnx(qat_model=self.imagenet_model, inputs=self.dummy_inputs, output_file=file_path, device=self.device_train)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ We have to apply the QAT scaffold before we load a checkpoint (if QAT is enabled),
            because Pytorch doesn't serialize the graph, just the params. Pytorch changesf
            all the param names when scaffolding is applied, so the state_dict will have 
            mismatching keys unless we scaffold the model first.
        """
        if self.use_qat:
            self._prepare_qat()
        return super().on_load_checkpoint(checkpoint)

