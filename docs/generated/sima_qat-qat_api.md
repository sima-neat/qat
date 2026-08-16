# `sima_qat.qat_api`

Source: `sima_qat/qat_api.py`

Public lifecycle API for the Dynamo-free SiMa FX QAT backend.

## Public API

### Function: `sima_prepare_qat_model(input_graph: nn.Module, inputs: Tuple[Any, ...], device: Union[str, torch.device]) -> GraphModule`

Prepare a symbolically traceable model for signed-int8 FX QAT.

This path uses FX graph-mode quantization only. It does not import or call
Dynamo, torch.export, PT2E prepare, or PT2E conversion APIs.

### Function: `sima_finalize_qat_model(qat_model: GraphModule) -> GraphModule`

Freeze observers and produce an inference-only fake-quantized model.

### Function: `sima_export_onnx(qat_model: nn.Module, inputs: Tuple[Any, ...], output_file: str, input_names: Optional[List[str]] = None, output_names: Optional[List[str]] = None, device: Optional[Union[str, torch.device]] = None) -> GraphModule`

Export a finalized QAT model as a standard opset-17 ONNX Q/DQ graph.
