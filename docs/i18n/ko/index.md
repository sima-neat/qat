---
title: 양자화 인식 학습
sidebar_label: 양자화 인식 학습
sidebar_position: 1
---

# 양자화 인식 학습

양자화 인식 학습(QAT)은 INT8 추론의 수치적 영향을 시뮬레이션하면서 PyTorch 모델을
미세 조정합니다. 학습 후 양자화로 인해 허용할 수 없는 정확도 손실이 발생하고 대표적인
데이터로 모델을 다시 학습할 수 있을 때 사용합니다.

기존 PyTorch 학습 프로젝트에 SiMa QAT를 추가하십시오. 데이터 세트, 전처리,
데이터 증강, 손실, 옵티마이저 설정 및 검증 지표는 유지하십시오.

가능하면 사전 학습된 부동 소수점 체크포인트에서 시작하십시오. 무작위 초기화부터 학습하는
것도 지원하지만 일반적으로 훨씬 더 많은 시간과 데이터가 필요합니다.

## QAT 작동 방식

준비 단계에서는 모델에 옵저버와 가짜 양자화 연산을 추가합니다. 옵저버는 활성화 범위를
측정하고, 가짜 양자화는 순전파 중 값을 반올림하고 제한하여 INT8 실행을 근사합니다.
텐서, 그래디언트 및 옵티마이저 업데이트는 부동 소수점으로 유지되므로 학습 과정에서
양자화 영향에 맞게 모델 가중치를 조정할 수 있습니다.

워크플로는 다음과 같습니다.

1. eager PyTorch 모델을 QAT용으로 **준비**합니다.
2. 일반 학습 루프로 옵저버를 **워밍업**합니다.
3. 활성화 범위와 SiMa 호환 가중치 스케일을 **고정**합니다.
4. 고정된 스케일로 학습을 계속하여 정확도를 **회복**합니다.
5. 추론용 모델로 **최종화**합니다.
6. `QuantizeLinear`와 `DequantizeLinear`(QDQ) 노드가 포함된 표준 opset-17
   ONNX 모델을 **내보냅니다**.

QDQ 노드는 내보낸 ONNX 그래프에서 부동 소수점 값과 INT8 값 사이의 변환을
표현합니다.

학습과 검증에서는 모델과 배치를 같은 디바이스에 두고, 가능하면 CUDA를 사용하십시오.
최종화는 CPU 모델을 반환하며, 내보내기는 전달된 모델의 디바이스를 유지합니다.

## 설치

QAT wheel에는 Python 3.10 이상과 PyTorch 2.8.x가 필요합니다. 모델의 학습 종속성이
이미 포함된 환경에 설치하십시오.

`sima-cli`를 사용하여 QAT 패키지를 다운로드합니다.

```bash
sima-cli neat install qat
```

이 명령은 wheel을 다운로드하고 Codex 및 Claude용 QAT 코딩 에이전트 스킬을 설치하거나
업데이트합니다. 활성 Python 환경은 변경하지 않습니다. 학습 환경을 활성화한 다음
다운로드한 wheel을 설치합니다.

```bash
python -m pip install ./sima_qat-*.whl
python -c "import torch, sima_qat; print(torch.__version__, sima_qat.__file__)"
```

## 학습 프로젝트에 QAT 추가

다음 단계는 하나의 연속된 워크플로를 구성합니다. 기존 학습 프로젝트에 맞게 모델, 데이터,
옵티마이저, 손실 및 검증 호출을 조정하십시오.

### 1. 모델 준비

옵티마이저를 만들기 전에 모델을 준비합니다. 준비 과정은 독립된 QAT 그래프를 반환하며
원본 모델이나 예제 입력을 수정하거나 이동하지 않습니다. 입력 튜플은 모델의 위치 인수,
데이터 타입 및 형상과 일치해야 합니다. `device` 인수는 반환되는 QAT 그래프의 학습
디바이스를 선택합니다. 가능한 경우 CUDA GPU를 사용하십시오.

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

학습은 준비된 그래프를 업데이트하므로 `source_model`이 아니라 `qat_model`의 매개변수로
옵티마이저를 만드십시오.

### 2. 학습, 고정 및 회복

워밍업 중에는 옵저버가 활성화 범위를 측정하고 가짜 양자화도 이미 INT8을 시뮬레이션합니다.
고정은 범위와 스케일 업데이트를 멈추며, **가중치 학습은 계속됩니다**. 고정된 설정에서
정확도를 회복할 수 있도록 이후에도 학습을 계속하십시오.

검증 중에는 가짜 양자화를 유지하고 옵저버를 비활성화하여 검증 데이터가 측정 범위를
바꾸지 않도록 해야 합니다. `eval()`만으로는 옵저버가 멈추지 않습니다. 이 분류용 헬퍼는
고정 단계에서 이미 비활성화된 옵저버를 포함하여 이전 모드와 상태를 복원합니다.
탐지 등 다른 작업에서는 지표를 바꾸십시오.

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

이 예제는 워밍업 2 에포크와 회복 2 에포크를 사용합니다. 이를 시작점으로 삼고, 검증
결과에 따라 고정 에포크와 회복 기간을 선택하십시오.

### 3. 학습 저장 및 재개

학습을 재개할 수 있도록 최종화 전에 준비된 모델을 저장합니다. 체크포인트에는 QAT 모델과
옵티마이저 상태가 모두 포함되어야 합니다.

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

학습을 재개하려면 저장된 상태를 불러오기 전에 동일한 예제 입력과 배치 조건으로 동일한
모델을 다시 만들고 준비합니다.

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

체크포인트는 옵저버 상태, 고정 상태, 최종화 및 ONNX 내보내기에 사용될 정확한 양자화
매개변수를 보존합니다.

### 4. 최종화 및 내보내기

학습 체크포인트를 먼저 저장하십시오. 최종화된 모델은 추론 전용이며 학습을 재개할 수
없습니다. 수동으로 CPU로 이동할 필요는 없습니다.

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

| 작업 | 디바이스 동작 |
|---|---|
| 준비 | 선택한 학습 디바이스의 QAT 모델을 반환하며 원본 모델은 변경하지 않습니다. |
| 최종화 | 전달된 QAT 모델을 CPU로 이동하고 CPU의 추론 전용 모델을 반환합니다. |
| 내보내기 | 일시적으로 CPU를 사용한 뒤 실패해도 모델 디바이스와 그래프의 디바이스 설정을 복원합니다. 예제 입력은 변경하지 않습니다. |

일반적으로 내보내기의 `device` 인수를 생략하십시오. `device="cuda"` 또는
`device="cpu"`를 명시하면 성공 후 모델을 해당 디바이스로 이동합니다. 이 옵션은
PyTorch 모델의 디바이스를 제어하며 ONNX 실행 디바이스를 지정하지 않습니다.

## 배치 크기

준비 과정은 기본적으로 첫 번째 텐서 차원(배치 차원)을 동적으로 유지합니다. 따라서 하나의 준비된
그래프로 일반적인 데이터 로더 배치 크기를 사용해 학습하고, 마지막의 짧은 배치를 처리하며,
`sima_export_onnx`에 전달된 구체적인 배치 크기로 내보낼 수 있습니다.

대부분의 모델에는 배치 옵션이 필요하지 않습니다. 모델이 예제와 정확히 같은 배치 크기를
의도적으로 요구할 때만 `dynamic_batch=False`를 설정하십시오.

```python
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
    dynamic_batch=False,
)
```

모델이 배치 크기를 검증하거나 분기 조건으로 사용하고, 고정 크기 순환 상태를 사용하거나,
배치를 방향, 채널 또는 다른 레이아웃 구조에 결합하는 경우 고정 배치 캡처가 적합합니다.
동적 캡처는 제공된 예제에서 캡처된 출력을 원본 모델과 비교하며, 동작이 바뀌는 경우
동적 배치를 비활성화하라는 안내와 함께 실패합니다.

## 검증 및 컴파일

원본 부동 소수점 모델, 고정 전후의 준비된 모델, 최종화된 PyTorch 모델 및 ONNX 모델의
작업 수준 지표를 측정하십시오. 이를 통해 어느 수명 주기 단계에서 회귀가 발생했는지
명확하게 확인할 수 있습니다.

컴파일하기 전에 내보낸 파일을 검증합니다.

```bash
python - <<'PY'
import onnx

model = onnx.load("model.qdq.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid")
PY
```

대표적인 검증 샘플에서 최종화된 PyTorch 모델과 ONNX Runtime을 비교하십시오. 양자화
경계에서 작은 요소별 차이가 발생할 수 있으므로 출력에 적합한 허용 오차를 사용하고
모델의 실제 정확도 지표를 확인하십시오.

일반적인 가중 연산, 활성화, 정규화, 풀링, 리덕션 및 형상/레이아웃 연산을 지원합니다.
`ArgMax`와 `TopK`의 인덱스 출력은 정수로 유지됩니다. PReLU, ConvTranspose,
Embedding/Gather, GridSample, ReduceMin 및 CumSum은 학습할 수 있지만 이번 릴리스에서는
QAT 어노테이션 대상이 아닙니다.

내보낸 QDQ ONNX 모델을 CPU의 Model Compiler에 전달하십시오. 가져오기, 파티셔닝,
최적화 및 하드웨어 할당은 별도의 컴파일 단계입니다.

## 실행 가능한 예제

저장소의 [examples](https://github.com/sima-neat/qat/tree/main/examples)에는 CPU에서 실행하기
쉬운 소규모 MNIST 워크플로, 사전 학습된 분류기의 ImageNet 미세 조정, 체크포인트 재개와
내보내기를 지원하는 순수 PyTorch YOLO26n 워크플로가 포함되어 있습니다.
