---
title: 量子化を考慮した学習
sidebar_label: 量子化を考慮した学習
sidebar_position: 1
---

# 量子化を考慮した学習

量子化を考慮した学習（QAT）は、INT8 推論の数値的な影響をシミュレートしながら
PyTorch モデルを微調整します。学習後量子化による精度低下が許容できず、
代表的なデータを使用してモデルを再学習できる場合に使用します。

既存の PyTorch 学習プロジェクトに SiMa QAT を追加します。データセット、前処理、
データ拡張、損失、オプティマイザの設定、検証指標は維持してください。

可能な場合は、事前学習済みの浮動小数点チェックポイントから開始してください。
ランダムな初期値からの学習も可能ですが、通常は大幅に多くの時間とデータが必要です。

## QAT の仕組み

準備処理では、モデルにオブザーバと疑似量子化演算を追加します。オブザーバは活性化範囲を
測定し、疑似量子化はフォワードパス中の値を丸めてクランプすることで INT8 実行を近似します。
テンソル、勾配、オプティマイザの更新は浮動小数点のままなので、量子化の影響に合わせて
モデルの重みを学習できます。

ワークフローは次のとおりです。

1. Eager モードの PyTorch モデルを QAT 用に **準備** します。
2. 通常の学習ループでオブザーバを **ウォームアップ** します。
3. 活性化範囲と SiMa 互換の重みスケールを **固定** します。
4. 固定したスケールで学習を続け、精度を **回復** します。
5. 推論用にモデルを **確定** します。
6. `QuantizeLinear` と `DequantizeLinear`（QDQ）ノードを含む標準の
   opset-17 ONNX モデルを **エクスポート** します。

QDQ ノードは、エクスポートされた ONNX グラフ内で浮動小数点値と INT8 値の
変換を表します。

学習と検証ではモデルとバッチを同じデバイスに置き、利用可能なら CUDA を使用します。
確定処理は CPU モデルを返します。エクスポートは渡されたモデルのデバイスを維持します。

## インストール

QAT wheel には Python 3.10 以降と PyTorch 2.8.x が必要です。モデルの学習依存関係が
すでに含まれている環境にインストールしてください。

`sima-cli` で QAT パッケージをダウンロードします。

```bash
sima-cli neat install qat
```

このコマンドは wheel をダウンロードし、Codex と Claude 用の QAT コーディングエージェント
スキルをインストールまたは更新します。現在の Python 環境は変更しません。
学習環境を有効にして、ダウンロードした wheel をインストールします。

```bash
python -m pip install ./sima_qat-*.whl
python -c "import torch, sima_qat; print(torch.__version__, sima_qat.__file__)"
```

## 学習プロジェクトへの QAT の追加

以下の手順は、一連のワークフローを構成します。既存の学習プロジェクトに合わせて、
モデル、データ、オプティマイザ、損失、検証処理を調整してください。

### 1. モデルを準備する

オプティマイザを作成する前にモデルを準備します。準備処理は独立した QAT グラフを返し、
元のモデルや入力例を変更したり移動したりしません。入力タプルは、モデルの位置引数、
データ型、形状と一致する必要があります。`device` 引数は返される QAT グラフの
学習デバイスを選択します。利用可能であれば CUDA GPU を使用してください。

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

学習では準備済みグラフが更新されるため、`source_model` ではなく `qat_model` から
オプティマイザを作成してください。

### 2. 学習、固定、回復

ウォームアップ中は、オブザーバが活性化範囲を測定し、偽量子化もすでに INT8 を
シミュレートしています。固定するのは範囲とスケールの更新であり、**重みの学習では
ありません**。その後も学習を続け、固定された設定で精度を回復させます。

検証では偽量子化を有効に保ち、オブザーバを無効にして検証データによる範囲の変更を
防ぎます。`eval()` だけではオブザーバは停止しません。この分類用ヘルパーは、固定により
無効になっているオブザーバも含め、以前のモードと状態を復元します。検出などの
タスクでは指標を変更してください。

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

この例ではウォームアップと回復にそれぞれ 2 エポックを使用します。開始点として考え、
検証結果に基づいて固定するエポックと回復期間を選んでください。

### 3. 学習の保存と再開

学習を再開できるように、確定処理の前に準備済みモデルを保存します。チェックポイントには
QAT モデルとオプティマイザの両方の状態を含めます。

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

再開するには、保存した状態を読み込む前に、同じ入力例とバッチ契約を使用して同じモデルを
再作成し、準備します。

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

チェックポイントには、オブザーバの状態、固定状態、確定処理と ONNX エクスポートで
使用される正確な量子化パラメータが保持されます。

### 4. 確定とエクスポート

先に学習用チェックポイントを保存してください。確定済みモデルは推論専用で、
学習を再開できません。手動で CPU に移す必要はありません。

```python
final_model = sima_finalize_qat_model(qat_model)
accuracy = validate(final_model, validation_loader, "cpu")
sima_export_onnx(
    final_model,
    example_inputs,
    "model.qdq.onnx",
    input_names=["images"],
    output_names=["predictions"],
)
```

| 操作 | デバイスの動作 |
|---|---|
| 準備 | 選択した学習デバイス上の QAT モデルを返します。元のモデルは変更しません。 |
| 確定 | 渡された QAT モデルを CPU に移し、CPU 上の推論専用モデルを返します。 |
| エクスポート | 一時的に CPU を使用し、失敗時もモデルのデバイスとグラフのデバイス設定を復元します。入力例は変更しません。 |

通常はエクスポートの `device` 引数を省略してください。`device="cuda"` または
`device="cpu"` を明示すると、成功後にモデルをそのデバイスへ移します。
このオプションは PyTorch モデル用であり、ONNX の実行デバイスを指定しません。

## バッチサイズ

準備処理では、テンソルの先頭次元（バッチ次元）をデフォルトで動的に保持します。これにより、
同じ準備済みグラフで通常のデータローダーのバッチサイズを使用し、最後の短いバッチを処理し、
`sima_export_onnx` に渡した具体的なバッチサイズでエクスポートできます。

ほとんどのモデルではバッチオプションは不要です。モデルが入力例と同じバッチサイズを
意図的に必要とする場合に限り、`dynamic_batch=False` を設定します。

```python
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
    dynamic_batch=False,
)
```

モデルがバッチサイズをアサートまたは分岐条件に使用する場合、固定サイズの再帰状態を
使用する場合、あるいはバッチを方向、チャネル、その他のレイアウト形状に組み込む場合は、
固定バッチのキャプチャが適しています。動的キャプチャは、指定された入力例に対する出力を
元のモデルと比較し、動作が変わる場合は動的バッチを無効にするための案内とともに失敗します。

## 検証とコンパイル

元の浮動小数点モデル、固定前後の準備済みモデル、確定済み PyTorch モデル、ONNX モデルに
ついて、タスクレベルの指標を測定します。これにより、どのライフサイクル手順で回帰が
発生したかを明確にできます。

コンパイルの前に、エクスポートしたファイルを検証します。

```bash
python - <<'PY'
import onnx

model = onnx.load("model.qdq.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid")
PY
```

代表的な検証サンプルで、確定済み PyTorch モデルと ONNX Runtime を比較します。
量子化境界では要素単位の小さな差が生じる可能性があるため、出力に適した許容誤差を使用し、
モデル本来の精度指標も確認してください。

一般的な重み付き演算、活性化、正規化、プーリング、リダクション、形状／レイアウト演算を
サポートしています。`ArgMax` と `TopK` のインデックス出力は整数のままです。PReLU、
ConvTranspose、Embedding/Gather、GridSample、ReduceMin、CumSum は学習可能ですが、
このリリースでは QAT アノテーションの対象外です。

エクスポートした QDQ ONNX モデルを CPU 上の Model Compiler に渡します。インポート、
パーティショニング、最適化、ハードウェア割り当ては、別のコンパイル手順です。

## 実行可能な例

リポジトリの [examples](https://github.com/sima-neat/qat/tree/main/examples) には、
CPU で実行しやすい小規模な MNIST ワークフロー、事前学習済み分類器の ImageNet 微調整、
チェックポイントの再開とエクスポートに対応した純粋な PyTorch の YOLO26n ワークフローが
含まれています。
