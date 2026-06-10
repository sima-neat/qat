# ImageNet Classifer example 

This folder contains the following helper files:
```
imagenet/
├── README.md
├── imagenet_lit.py
├── train.py
├── test_onnx.py
└── export_onnx.py
```

This example can be used to train any of classifiers available on Torchvision with the ImageNet dataset. 

## File Descriptions

- **README.md** : Provides an overview of the example, file structure and descriptions, instructions on usage.
- **imagenet_lit.py** : Contains the main Lightning module implementation for training and validating the ImageNet Torchvision model using PyTorch Lightning.
- **train.py**: Script to train the ImageNet Torchvision model. It leverages the Lightning module defined in imagenet_lit.py.
- **test_onnx.py** : Script to test the ONNX model. This evaluates the exported ONNX model's performance on the test set.
- **export_onnx.py** : Script to export the the last trained Pytorch checkpoint into the ONNX format, which can be used for running inference on different platforms supporting ONNX.


## Train ImageNet Torchvision model using QAT
The training process for the ImageNet Torchvision model uses PyTorch Lightning to streamline the training, checkpointing, and model export. Below is an overview of the key steps involved:

* Dataset Preparation - 
Before using this example, it is required to download ImageNet 2012 dataset in a directory and place the split files under the respective train, test, val folders. 

* Training the Model - 
The training is managed by PyTorch Lightning's Trainer class, which simplifies the training loop, logging, and checkpointing.
The script `train.py` starts the training process, leveraging the `imagenet_lit.py` pyTorch lightning module, which defines the model, training, validation steps, and metrics.

* QAT User API Usage -
The user can directly leverage the `imagenet_lit.py` pyTorch lightning module which internally calls the QAT user APIs. 
    - The `on_train_start()` hook calls the `sima_prepare_qat_model()` which any Pytorch nn.Module and prepares it for QAT training. 
    - The `on_train_end()` hook calls the `sima_finalize_qat_model()` which takes a trained QAT model and converts it to a quantized model. It becomes inference-only after this point. 
    - The `on_fit_end()` hook calls the `sima_export_onnx()` which finalized QAT model and exports a ONNX graph for the same.

* Training Script Arguments - 
The training process can be customized using various command-line arguments defined in `train.py`. These arguments allow you to control key aspects of the training, such as the number of epochs, batch size, dataset location, and more. 

The Arguments can be found using the `--help` command as below:
`python train.py --help`

Below are the descriptions of the arguments:

| Argument                | Default Value | Description                                                                                                           | Example Usage                     |
|-------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `-e, --epochs`          | `10`          | The number of epochs to train the model (i.e., how many times the entire dataset is passed through the model).       | `--epochs 20`                     |
| `-b, --batch`           | `16`          | Specifies the batch size, which is the number of training samples used in each training iteration. NOTE: Increase the batch size accordingly, as lower batch size may take a really long time to train.                  | `--batch 32`                      |
| `-d, --data`            | `"."`         | The path where the dataset is located. If not present, the dataset can be downloaded to this path with `--download`. | `--data /path/to/dataset`        |
| `--model`            | `resnet18`           | The torchvision Imagenet Model to be trained, the pre-trained model is loaded from torchvision.                                      | `--model model_name`                      |
| `--device`              | `"cpu"`       | The device to use for training: `"mps"` for Apple Silicon GPUs, `"cuda"` for NVIDIA GPUs, or `"cpu"` for CPU.      | `--device cuda`                   |
| `--samples-limit`       | `1281167`       | Limits the number of training samples used. Useful for testing or debugging with a smaller dataset.                  | `--samples-limit 10000`          |
| `--export-on-end`       | `False`           | Export the trained model to ONNX format at the end of training.                                                     | `--export-on-end`                 |
| `--disable-qat`         | `False`           | Disable Quantization Aware Training (QAT), which prepares the model for quantization during training.               | `--disable-qat`                   |
| `--resume`              | `False`          | Resume training from the most recent checkpoint if available, allowing for interrupted training sessions to continue. | `--resume`                        |

* Example Usage - 
`python train.py --export-on-end -b 100 --device cuda -d /data/imagenet`

* Checkpoints - 
During training, the script automatically saves checkpoints at various stages to allow for model recovery and resuming training. These checkpoints are stored in the **checkpoints/** directory by default. 

* ONNX Model Export -
Once training is complete, the trained model is exported to the ONNX format for compatibility with various inference engines and platforms.
The script handles this, exporting the final model as `exported_model.onnx`, which is saved in the project directory or a specified location. This format allows for easy deployment in environments that support ONNX.


## Test QAT ONNX model
The `test_onnx.py` script is used to test a trained ImageNet torchvision model saved in the ONNX format.  

Below are the command-line arguments for this script:

| Argument                | Default Value       | Description                                                                                                    | Example Usage                        |
|-------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------|
| `--onnx`                | `recent_onnx_file`  | The path to the ONNX file containing the trained ImageNet Torchvision model. The script finds the most recent onnx file and sets it to the name `recent_onnx_file`.                                                 | `--onnx /path/to/model.onnx`         |
| `--dsroot`              | `.`                 | The root directory of the dataset, used for testing the model.                                                 | `--dsroot /path/to/dataset`          |                       |
| `--split`              | `val`                 | Dataset split (test or val) to be used for testing.                                                 | `--split val`          |                       |
| `-v, --verbosity`       | `INFO`              | Sets the logging verbosity level (e.g., DEBUG, INFO, WARNING, ERROR).                                          | `--verbosity DEBUG`                  |

The Arguments can also be found using the `--help` command as below:
`python test_onnx.py --help`

* Example Usage - 
`python test_onnx.py --dsroot /data/imagenet` 


## Export any trained checkpoint to ONNX
The `export_onnx.py` script is used to export the most recent checkpoint of the trained model to an ONNX file. 

Below are the command-line arguments for this script:

| Argument                | Default Value       | Description                                                                                                      | Example Usage                        |
|-------------------------|---------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| `--model`            | required       | Model name to search for in checkpoint files.                                      | `--model model_name`    |
| `-c, --ckpt`            | `latest_ckpt`       | The path to the checkpoint file to be loaded and exported as an ONNX model.                                      | `--ckpt /path/to/checkpoint.ckpt`    |
| `--device`              | `cpu`               | The device to use for exporting the model. Options include `"cpu"`, `"cuda"`, `"mps"` for Apple GPUs, etc.        | `--device cuda`                      |

The Arguments can also be found using the `--help` command as below:
`python export_onnx.py --help`

* Example Usage - 
`python export_onnx.py --model resnet18`


## DEBUGGING TIPS
- In case you want to check any device related information or take a look at the QAT scaffolded model or the finalized QAT model and your model is in the form of an torch fx graph, you can use the `{fx_graph_model}.graph.print_tabular()` utility function which displays information about the model nodes and arguments. 
