---
title: 量化感知訓練
sidebar_label: 量化感知訓練
sidebar_position: 1
---

# 量化感知訓練

量化感知訓練（QAT）會在模擬 INT8 推論數值效果的同時微調 PyTorch 模型。
當訓練後量化造成無法接受的準確度損失，而且您可以使用具代表性的資料重新訓練模型時，
請使用 QAT。

將 SiMa QAT 加入現有的 PyTorch 訓練專案。保留資料集、前處理、資料增強、損失函數、
最佳化器設定與驗證指標。

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

QDQ 節點描述匯出 ONNX 圖中浮點值與 INT8 值之間的轉換。

訓練與驗證時，模型和批次資料應位於相同裝置；可用時請使用 CUDA。完成程序維持模型的
原本裝置。匯出會暫時使用 CPU，之後還原傳入模型的裝置。

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
`device` 引數會選擇傳回 QAT 圖的訓練裝置；如有可用的 CUDA GPU，請優先使用。

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

預熱時，觀察器會測量啟用值範圍，而假量化也已開始模擬 INT8。凍結會停止範圍與
縮放因子的更新，**不會停止權重訓練**。之後請繼續訓練，讓權重在固定設定下恢復準確度。

驗證時必須保留假量化並停用觀察器，避免驗證資料改變測量範圍。僅使用 `eval()`
不會停止觀察器。這個分類輔助函式會還原先前的模式與觀察器狀態，包括凍結時已停用
的觀察器。偵測或其他任務請調整指標。

```python
from torch.ao.quantization import disable_observer
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

def validate(model, loader, device):
    was_training = model.training
    observer_states = [
        (module, module.observer_enabled.clone())
        for module in model.modules()
        if isinstance(module, FakeQuantizeBase)
    ]
    correct = total = 0
    try:
        model.eval()
        model.apply(disable_observer)
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()
    finally:
        model.train(was_training)
        for module, enabled in observer_states:
            module.observer_enabled.copy_(enabled)
    return correct / total

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

    accuracy = validate(qat_model, validation_loader, device)
    print(f"Epoch {epoch}: validation accuracy {accuracy:.2%}")
```

此範例使用兩個預熱 epoch 與兩個恢復 epoch。請將它視為起點，依驗證結果選擇
凍結的 epoch 與恢復訓練的時間。

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

請先儲存訓練檢查點：完成後的模型僅供推論，無法繼續訓練。不需要手動移至 CPU。

```python
final_model = sima_finalize_qat_model(qat_model)
accuracy = validate(final_model, validation_loader, device)
sima_export_onnx(
    final_model,
    example_inputs,
    "model.qdq.onnx",
    input_names=["images"],
    output_names=["predictions"],
)
```

| 操作 | 裝置行為 |
|---|---|
| 準備 | 傳回所選訓練裝置上的 QAT 模型；原始模型維持不變。 |
| 完成 | 傳回與 QAT 模型位於相同裝置、僅供推論使用的模型。 |
| 匯出 | 暫時使用 CPU，之後還原模型裝置與圖中的裝置設定，即使失敗也會還原。範例輸入維持不變。 |

通常可省略匯出的 `device` 引數。明確指定 `device="cuda"` 或 `device="cpu"`，會在
匯出成功後將模型移至該裝置。這個選項控制 PyTorch 模型，而非 ONNX 的執行裝置。

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

將匯出的 QDQ ONNX 模型交給 CPU 上的 Model Compiler。匯入、分割、最佳化與硬體
指派屬於獨立的編譯步驟。

## 可執行的範例

儲存庫的 [examples](https://github.com/sima-neat/qat/tree/main/examples) 包含適合 CPU 的
小型 MNIST 工作流程、預先訓練分類器的 ImageNet 微調，以及支援檢查點續訓與匯出的
純 PyTorch YOLO26n 工作流程。
