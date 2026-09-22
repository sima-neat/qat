from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import optim, nn, utils, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from torch.fx.graph_module import GraphModule

import pytorch_lightning as L

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


class MNIST_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output


class MNIST_Trainer(L.LightningModule):
    def __init__(
        self,
        use_qat: bool = True,
        export_on_end: bool = False,
        freeze_epoch: int | None = None,
    ):
        super().__init__()
        self.mnist_model = MNIST_Model()
        self.loss_fn = CrossEntropyLoss()
        self.prev_epoch_step = 0
        self.val_accuracy = 0
        self.val_batch_count = 0
        self.use_qat = use_qat
        self.export_on_end = export_on_end
        self.freeze_epoch = freeze_epoch
        self.dump_fx_graphs = True
        self.dummy_inputs = (torch.randn(1, 1, 28, 28), )
        # Call this last once all init has been done
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.use_qat:
            # Preparation replaces the eager parameters with captured QAT
            # parameters, so it must happen before the optimizer is built.
            self._prepare_qat()
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.mnist_model(imgs)
        
    def _step(self, batch, batch_idx):
        x, gt = batch
        logits_y = self.mnist_model(x)
        loss = self.loss_fn(logits_y, gt)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, gt = batch
        logits_y = self.mnist_model(x)
        loss = self.loss_fn(logits_y, gt)
        # We run the validation set here.
        self.log("val_loss", loss, prog_bar=True)
        # Compute top-1 accuracy here and keep a running tab of results.
        scores = torch.argmax(logits_y, dim=-1) == gt
        self.val_accuracy += torch.mean(scores.type(torch.float32))
        self.val_batch_count += 1
        return loss

    def on_validation_end(self) -> None:
        super().on_validation_end()
        top1_acc = self.val_accuracy / self.val_batch_count
        print(f"Validation top-1 accuracy: {top1_acc}")
        self.val_accuracy = 0
        self.val_batch_count = 0
        self.prev_epoch_step = self.global_step 

    def on_train_start(self) -> None:
        super().on_train_start()
        if not self.use_qat:
            # Do a compile so we can see an FX graph
            print(f"Compiling model to FX graph ...")
            self._dump_fx_graph('compiled_graph.txt')

    def on_train_end(self) -> None:
        super().on_train_end()
        self._finalize_qat_model()

    def on_train_epoch_start(self) -> None:
        # For some reason Lightning doesn't switch to train mode hence, we ensure it switches to train mode here
        self.train(True)
        if (
            self.use_qat
            and self.freeze_epoch is not None
            and self.current_epoch == self.freeze_epoch
        ):
            sima_freeze_qat(self.mnist_model)

    def _prepare_qat(self) -> None:
        m = sima_prepare_qat_model(input_graph=self.mnist_model, inputs=self.dummy_inputs, device=self.device)
        # Now replace our model
        setattr(self, 'mnist_model', m)
        self._dump_fx_graph('prepare_p2e_graph.txt')

    def _finalize_qat_model(self) -> None:
        self.train(False)
        # If we are running in QAT mode, we first convert to a quantized graph.
        if self.use_qat:
            m = sima_finalize_qat_model(self.mnist_model)
            # Now replace our model
            setattr(self, 'mnist_model', m)
            self._dump_fx_graph('post_p2e_graph.txt')
        return

    def on_fit_end(self) -> None:
        if self.export_on_end:
            self.to_onnx(file_path='exported_model.onnx')

    def _dump_fx_graph(self, fname: str) -> None:
        if not self.dump_fx_graphs:
            return
        if not isinstance(self.mnist_model, GraphModule):
            from torch.fx import symbolic_trace
            # Symbolic tracing frontend - captures the semantics of the module
            symbolic_traced : torch.fx.GraphModule = symbolic_trace(self.mnist_model)
            # If the graph is not already in FX format, we skip this entire step.
            # return
        else:
            symbolic_traced = self.mnist_model

        # High-level intermediate representation (IR) - Graph representation
        print(f"Generating graph dump to file: {fname}")
        with open(fname, 'wt') as f:
            print(symbolic_traced.graph, file=f)

    def to_onnx(self, file_path: str | Path, input_sample: Any | None = None, **kwargs: Any) -> None:
        """ This function needs to be overridden in the case of QAT, since export gets tricky
            and specialized.
        """
        self.train(False)
        sima_export_onnx(qat_model=self.mnist_model, inputs=self.dummy_inputs, output_file=file_path)
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ We have to apply the QAT scaffold before we load a checkpoint (if QAT is enabled),
            because Pytorch doesn't serialize the graph, just the params. Pytorch changes
            all the param names when scaffolding is applied, so the state_dict will have 
            mismatching keys unless we scaffold the model first.
        """
        if self.use_qat:
            self._prepare_qat()
        return super().on_load_checkpoint(checkpoint)
