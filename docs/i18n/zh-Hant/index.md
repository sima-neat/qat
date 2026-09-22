---
title: 量化感知訓練
sidebar_label: 量化感知訓練
sidebar_position: 1
---

# 量化感知訓練

量化感知訓練（QAT）會在模擬 INT8 推論數值效果的同時微調 PyTorch 模型。
當訓練後量化造成無法接受的準確度損失，而且您可以使用具代表性的資料重新訓練模型時，
請使用 QAT。

SiMa QAT 的設計用途是在模型現有的 PyTorch 訓練專案中執行。它不會取代資料集、
資料增強、損失函數、最佳化器或驗證指標。保留原始專案的這些部分很重要，因為它們
通常正是浮點模型能夠達到良好準確度的原因。

盡可能從預先訓練的浮點檢查點開始。系統也支援從隨機初始化開始訓練，但通常需要
更多時間與資料。

## QAT 的運作方式

準備程序會將觀察器與假量化運算加入模型。觀察器會測量啟用值範圍，而假量化會在
正向傳遞期間對數值進行捨入與限幅，以近似 INT8 執行。張量、梯度和最佳化器更新仍
保持浮點格式，因此訓練可以調整模型權重以適應量化效果。

工作流程如下：

1. 為 QAT **準備** eager PyTorch 模型。
2. 使用一般訓練迴圈**預熱**觀察器。
3. **凍結**啟用值範圍與 SiMa 相容的權重縮放因子。
4. 使用鎖定的縮放因子繼續訓練，以**恢復**準確度。
5. **完成**用於推論的模型。
6. **匯出**包含 `QuantizeLinear` 與 `DequantizeLinear`（QDQ）節點的標準
   opset-17 ONNX 模型。

## 安裝

QAT wheel 需要 Python 3.10 或更新版本以及 PyTorch 2.8.x。請將它安裝到已包含
模型訓練相依套件的環境中。

使用 `sima-cli` 下載 QAT 套件：

```bash
sima-cli neat install qat
```

此命令會下載 wheel，並安裝或更新供 Codex 與 Claude 使用的 QAT 程式碼 Agent 技能。
它不會變更目前的 Python 環境。請啟用訓練環境，然後安裝下載的 wheel：

```bash
python -m pip install ./sima_qat-*.whl
python -c "import torch, sima_qat; print(torch.__version__, sima_qat.__file__)"
```

## 將 QAT 加入訓練專案

下列步驟組成一個連續的工作流程。請依現有訓練專案調整模型、資料、最佳化器、
損失函數與驗證呼叫。

### 1. 準備模型

請在建立最佳化器之前準備模型。準備程序會傳回獨立的 QAT 圖，不會修改或移動
來源模型或範例輸入。輸入 tuple 必須符合模型的位置輸入、資料類型與形狀。

```python
import torch
from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model and load the floating-point checkpoint.
source_model = MyModel()
source_model.load_state_dict(torch.load("model-fp32.pt", map_location="cpu"))
source_model.train()

example_inputs = (torch.randn(1, 3, 224, 224),)
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
)

optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
```

由於訓練會更新準備後的圖，請從 `qat_model` 而非 `source_model` 建立最佳化器。

### 2. 訓練、凍結與恢復

一開始請照常訓練，讓觀察器測量具代表性的啟用值範圍。完成預熱後凍結量化參數，
再繼續訓練，讓模型在鎖定的量化網格下恢復準確度。

```python
freeze_epoch = 2
num_epochs = 4

for epoch in range(num_epochs):
    qat_model.train()

    # Reserve one or more later epochs for recovery training.
    if epoch == freeze_epoch:
        sima_freeze_qat(qat_model)

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = qat_model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    validate(qat_model, validation_loader, device)
```

凍結 epoch 取決於模型。實用的起點是在短期微調的大部分時間預熱觀察器，並保留
至少最後一個 epoch 進行恢復訓練。請追蹤凍結前後的驗證準確度。如果準確度大幅下降，
請提早凍結並增加恢復訓練時間。

### 3. 儲存與繼續訓練

請在完成模型前儲存準備後的模型，以便繼續訓練。檢查點應同時包含 QAT 模型與
最佳化器狀態。

```python
from pathlib import Path

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "epoch": epoch,
        "model": qat_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    checkpoint_dir / f"qat-{epoch:02d}.pt",
)
```

若要繼續訓練，請使用相同的範例輸入與批次規格重新建立並準備相同模型，再載入
儲存的狀態：

```python
checkpoint = torch.load("checkpoints/qat-03.pt", map_location="cpu")

source_model = MyModel()
source_model.load_state_dict(torch.load("model-fp32.pt", map_location="cpu"))
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
)
optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5)

qat_model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1
```

檢查點會保留觀察器狀態、凍結狀態，以及稍後用於完成模型與 ONNX 匯出的精確
量化參數。

### 4. 完成與匯出

完成程序會建立僅供推論使用的模型。請將已訓練的 QAT 模型與範例輸入移至 CPU，
再將完成後的模型匯出為 opset-17 QDQ ONNX。

```python
final_model = sima_finalize_qat_model(qat_model.cpu())
export_inputs = tuple(value.cpu() for value in example_inputs)
sima_export_onnx(
    final_model,
    export_inputs,
    "model.qdq.onnx",
    input_names=["images"],
    output_names=["predictions"],
    device="cpu",
)
```

## 批次大小

準備程序預設會將張量的首個維度（即 Batch 維度）保持為動態。因此，同一個準備後的圖可以使用
一般資料載入器的批次大小進行訓練、處理最後一個較短的批次，並使用傳給
`sima_export_onnx` 的具體批次大小匯出。

大多數模型不需要批次選項。只有模型刻意要求與範例完全相同的批次大小時，才設定
`dynamic_batch=False`：

```python
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
    dynamic_batch=False,
)
```

當模型對批次大小進行斷言或依其分支、使用固定大小的循環狀態，或將批次合併到方向、
通道或其他版面配置幾何時，適合使用固定批次擷取。動態擷取會在提供的範例上比較
擷取結果與原始模型輸出；若行為改變，則會失敗並提示停用動態批次。

## 驗證與編譯

請針對原始浮點模型、凍結前後的準備模型、完成後的 PyTorch 模型，以及 ONNX 模型
測量工作層級的指標。這能清楚顯示是哪個生命週期步驟造成準確度下降（Regression）。

編譯之前，請驗證匯出的檔案：

```bash
python - <<'PY'
import onnx

model = onnx.load("model.qdq.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid")
PY
```

使用具代表性的驗證樣本，比較完成後的 PyTorch 模型與 ONNX Runtime。量化邊界可能
出現小幅逐元素差異，因此請使用適合輸出的容許誤差，並確認模型真正的準確度指標。

系統涵蓋常見的加權、啟用、正規化、池化、歸約和形狀／版面配置運算。
`ArgMax` 與 `TopK` 的索引輸出維持整數。PReLU、ConvTranspose、Embedding/Gather、
GridSample、ReduceMin 和 CumSum 仍可訓練，但此版本不會為它們加上 QAT 註解。

匯出的 QDQ ONNX 模型是交付 Model Compiler 的介面。匯入、分割、最佳化與硬體指派
屬於後續且獨立的編譯步驟。

## 可執行的範例

儲存庫的 [examples](https://github.com/sima-neat/qat/tree/main/examples) 包含適合 CPU 的
小型 MNIST 工作流程、預先訓練分類器的 ImageNet 微調，以及支援檢查點續訓與匯出的
純 PyTorch YOLO26n 工作流程。
