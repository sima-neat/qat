"""Versioned source of truth for the SiMa opset-17 INT8 QAT contract.

Compiler support is necessary but not sufficient for QAT support.  Each entry
therefore records the PyTorch capture contract, quantization behavior, and test
obligations independently of the compiler's ONNX operator database.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


OPERATOR_MANIFEST_VERSION = "1.0.0"
ONNX_OPSET = 17
COMPILER_OPERATOR_SOURCE = {
    "repository": "sima-neat/model-compiler",
    "pull_request": "https://github.com/sima-neat/model-compiler/pull/111",
    "revision": "7b9cf790aed8367f93ebd3a74064e200ac2fdc08",
    "schema_version": "4",
    "release": "2.1",
}


class SupportStatus(str, Enum):
    """Whether the declared PyTorch-to-QDQ contract is release-ready."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class QuantizationBehavior(str, Enum):
    """How an operator participates in the quantized graph."""

    ANNOTATION = "annotation"
    PROPAGATION = "propagation"
    TYPE_PRESERVING = "type-preserving"
    MIXED_OUTPUT = "mixed-output"
    DECOMPOSED = "decomposed"
    TRAINING_ONLY = "training-only"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class OperatorManifestEntry:
    family: str
    compiler_operator: str
    compiler_opset: int | None
    compiler_int8_supported: bool
    status: SupportStatus
    pytorch_modules: tuple[str, ...]
    functional_apis: tuple[str, ...]
    captured_aten_forms: tuple[str, ...]
    behavior: QuantizationBehavior
    supported_dtypes: tuple[str, ...]
    operand_constraints: tuple[str, ...]
    annotator: str | None
    additional_annotators: tuple[str, ...]
    expected_onnx_operators: tuple[str, ...]
    positive_test_requirements: tuple[str, ...]
    negative_test_requirements: tuple[str, ...]
    test_case_ids: tuple[str, ...] = ()
    onnx_test_case_ids: tuple[str, ...] = ()


W8A8 = (
    "torch.float32 training tensors",
    "signed INT8 activation QDQ",
    "signed INT8 weight QDQ where the operator has weights",
)
A8 = ("torch.float32 training tensors", "signed INT8 activation QDQ")
GRID = ("floating tensors carrying an existing signed INT8 activation grid",)
LIFECYCLE = (
    "capture and prepare",
    "real forward/backward update",
    "observer freeze",
    "finalization and execution",
)
ONNX_LIFECYCLE = LIFECYCLE + (
    "opset-17 ONNX checker",
    "QDQ placement and ONNX Runtime parity",
)
DEFAULT_NEGATIVE_TESTS = (
    "unsupported dtype, rank, axis, and operand forms do not receive misleading QAT annotations",
)


def _entry(
    family: str,
    compiler_operator: str,
    compiler_opset: int | None,
    *,
    status: SupportStatus = SupportStatus.SUPPORTED,
    compiler_int8: bool = True,
    modules: tuple[str, ...] = (),
    functions: tuple[str, ...] = (),
    aten: tuple[str, ...] = (),
    behavior: QuantizationBehavior = QuantizationBehavior.ANNOTATION,
    dtypes: tuple[str, ...] = A8,
    constraints: tuple[str, ...] = (),
    annotator: str | None = None,
    additional_annotators: tuple[str, ...] = (),
    onnx: tuple[str, ...] | None = None,
    positive: tuple[str, ...] = ONNX_LIFECYCLE,
    negative: tuple[str, ...] = DEFAULT_NEGATIVE_TESTS,
    cases: tuple[str, ...] = (),
    onnx_cases: tuple[str, ...] = (),
) -> OperatorManifestEntry:
    return OperatorManifestEntry(
        family=family,
        compiler_operator=compiler_operator,
        compiler_opset=compiler_opset,
        compiler_int8_supported=compiler_int8,
        status=status,
        pytorch_modules=modules,
        functional_apis=functions,
        captured_aten_forms=aten,
        behavior=behavior,
        supported_dtypes=dtypes,
        operand_constraints=constraints,
        annotator=annotator,
        additional_annotators=additional_annotators,
        expected_onnx_operators=onnx or (compiler_operator,),
        positive_test_requirements=positive,
        negative_test_requirements=negative,
        test_case_ids=cases,
        onnx_test_case_ids=onnx_cases,
    )


OPERATOR_MANIFEST = (
    _entry(
        "convolution",
        "Conv",
        11,
        modules=("torch.nn.Conv1d", "torch.nn.Conv2d"),
        functions=("torch.nn.functional.conv1d", "torch.nn.functional.conv2d"),
        aten=("aten.conv1d.default", "aten.conv2d.default"),
        dtypes=W8A8,
        constraints=("dilation 1..63", "stride 1..31", "static weights"),
        annotator="conv",
        additional_annotators=(
            "sima_conv_bn_hardtanh", "conv_bn_relu", "conv_bn",
            "sima_conv_add_or_mul_const", "sima_conv_hardtanh", "conv_relu",
            "sima_unannotated_conv2d",
        ),
        cases=(
            "conv1d", "conv2d", "reused_conv2d", "conv_relu", "conv_hardtanh",
            "conv_bn", "conv_bn_relu", "conv_bn_hardtanh",
            "conv_add_constant", "conv_mul_constant",
        ),
        onnx_cases=("conv2d", "conv_bn_relu"),
    ),
    _entry(
        "conv_transpose",
        "ConvTranspose",
        11,
        status=SupportStatus.DEFERRED,
        modules=("torch.nn.ConvTranspose2d",),
        functions=("torch.nn.functional.conv_transpose2d",),
        aten=("aten.conv_transpose2d.input",),
        dtypes=W8A8,
        constraints=(
            "dilation must be 1", "group=1 or depthwise", "output-channel weight axis is unresolved",
        ),
        behavior=QuantizationBehavior.UNSUPPORTED,
        negative=("preparation rejects the incorrect axis-0 weight contract",),
    ),
    _entry(
        "linear",
        "Gemm",
        13,
        modules=("torch.nn.Linear",),
        functions=("torch.nn.functional.linear",),
        aten=("aten.linear.default",),
        dtypes=W8A8,
        constraints=("rank-2 constant weight", "supported constant or runtime bias shapes"),
        annotator="linear",
        additional_annotators=("linear_relu",),
        cases=("linear", "linear_relu"),
        onnx_cases=("linear",),
    ),
    _entry(
        "matrix_multiply",
        "MatMul",
        13,
        functions=("torch.matmul", "operator.matmul", "torch.mm", "torch.bmm", "torch.baddbmm"),
        aten=("aten.matmul.default", "aten.mm.default", "aten.bmm.default", "aten.baddbmm.default"),
        constraints=("floating tensor operands", "BAddBMM bias is a floating tensor"),
        annotator="sima_matmul",
        cases=("mm", "matmul", "bmm", "baddbmm"),
        onnx_cases=("mm",),
        negative=("integer operands are not annotated",),
    ),
    _entry(
        "einsum",
        "Einsum",
        12,
        functions=("torch.einsum",),
        aten=("aten.einsum.default",),
        constraints=("exactly two operands", "each equation index follows compiler multiplicity constraints"),
        annotator="sima_einsum",
        cases=("einsum",),
        negative=("reject unsupported equations and non-floating operands",),
    ),
    _entry(
        "add",
        "Add",
        14,
        functions=("torch.add", "operator.add"),
        aten=("aten.add.Tensor",),
        constraints=("same shape, scalar, or compiler-supported broadcasting",),
        annotator="add",
        additional_annotators=("sima_add_hardtanh", "add_relu"),
        cases=("add", "add_relu", "add_hardtanh"),
        onnx_cases=("add",),
    ),
    _entry(
        "multiply",
        "Mul",
        14,
        functions=("torch.mul", "operator.mul"),
        aten=("aten.mul.Tensor",),
        constraints=("same shape, scalar, or compiler-supported broadcasting",),
        annotator="mul",
        additional_annotators=("mul_relu",),
        cases=("mul", "mul_relu"),
    ),
    _entry(
        "subtract",
        "Sub",
        14,
        functions=("torch.sub", "operator.sub"),
        aten=("aten.sub.Tensor",),
        constraints=("same shape, scalar, or compiler-supported broadcasting",),
        annotator="sima_binary_int8",
        cases=("sub",),
    ),
    _entry(
        "divide",
        "Div",
        14,
        functions=("torch.div", "operator.truediv"),
        aten=("aten.div.Tensor",),
        constraints=("same shape, scalar, or compiler-supported broadcasting", "nonzero denominator"),
        annotator="sima_binary_int8",
        cases=("div",),
        negative=("integer operands and zero-denominator test data are rejected",),
    ),
    _entry(
        "absolute",
        "Abs",
        13,
        functions=("torch.abs",),
        aten=("aten.abs.default",),
        annotator="sima_unary_int8",
        cases=("abs",),
    ),
    _entry("elu", "Elu", 6, modules=("torch.nn.ELU",), functions=("torch.nn.functional.elu",), aten=("aten.elu.default",), annotator="sima_unary_int8", cases=("elu",)),
    _entry("exponential", "Exp", 13, functions=("torch.exp",), aten=("aten.exp.default",), annotator="sima_unary_int8", cases=("exp",)),
    _entry("hard_sigmoid", "HardSigmoid", 6, modules=("torch.nn.Hardsigmoid",), functions=("torch.nn.functional.hardsigmoid",), aten=("aten.hardsigmoid.default",), annotator="sima_unary_int8", cases=("hardsigmoid",)),
    _entry("hard_swish", "HardSwish", 14, modules=("torch.nn.Hardswish",), functions=("torch.nn.functional.hardswish",), aten=("aten.hardswish.default",), annotator="sima_unary_int8", cases=("hardswish",)),
    _entry("leaky_relu", "LeakyRelu", 16, modules=("torch.nn.LeakyReLU",), functions=("torch.nn.functional.leaky_relu",), aten=("aten.leaky_relu.default",), annotator="sima_unary_int8", cases=("leaky_relu",)),
    _entry("logarithm", "Log", 13, functions=("torch.log",), aten=("aten.log.default",), constraints=("positive input domain",), annotator="sima_unary_int8", cases=("log",), negative=("exercise non-positive-domain diagnostics",)),
    _entry("negate", "Neg", 13, functions=("torch.neg", "operator.neg"), aten=("aten.neg.default",), annotator="sima_unary_int8", cases=("neg",)),
    _entry("reciprocal", "Reciprocal", 13, functions=("torch.reciprocal",), aten=("aten.reciprocal.default",), constraints=("nonzero input",), annotator="sima_unary_int8", cases=("reciprocal",)),
    _entry("softplus", "Softplus", 1, modules=("torch.nn.Softplus",), functions=("torch.nn.functional.softplus",), aten=("aten.softplus.default",), annotator="sima_unary_int8", cases=("softplus",)),
    _entry("square_root", "Sqrt", 13, functions=("torch.sqrt",), aten=("aten.sqrt.default",), constraints=("non-negative input domain",), annotator="sima_unary_int8", cases=("sqrt",)),
    _entry("tanh", "Tanh", 13, modules=("torch.nn.Tanh",), functions=("torch.tanh",), aten=("aten.tanh.default",), annotator="sima_unary_int8", cases=("tanh",)),
    _entry(
        "power",
        "Pow",
        15,
        functions=("torch.pow", "operator.pow"),
        aten=("aten.pow.Tensor_Scalar",),
        constraints=("constant scalar exponent in {0.5, -0.5, 2, 3}",),
        annotator="sima_pow",
        cases=("pow",),
        negative=("reject dynamic and unsupported exponents",),
    ),
    _entry(
        "prelu",
        "PRelu",
        16,
        modules=("torch.nn.PReLU",),
        functions=("torch.nn.functional.prelu",),
        aten=("aten.prelu.default",),
        constraints=("alpha is a 1D tensor using an activation-style per-tensor qspec",),
        annotator="sima_prelu",
        cases=("prelu",),
    ),
    _entry(
        "relu",
        "Relu",
        14,
        status=SupportStatus.PARTIAL,
        modules=("torch.nn.ReLU",),
        functions=("torch.relu", "torch.nn.functional.relu"),
        aten=("aten.relu.default",),
        behavior=QuantizationBehavior.PROPAGATION,
        dtypes=GRID,
        annotator="propagate_annotation",
        constraints=("currently covered after Conv, Linear, Add, and Mul",),
        cases=("conv_relu", "conv_bn_relu", "linear_relu", "add_relu", "mul_relu"),
        negative=("standalone and fan-out topology coverage is required",),
    ),
    _entry(
        "clip_hardtanh",
        "Clip",
        13,
        status=SupportStatus.PARTIAL,
        modules=("torch.nn.Hardtanh",),
        functions=("torch.clamp", "torch.nn.functional.hardtanh"),
        aten=("aten.hardtanh.default", "aten.hardtanh_.default"),
        behavior=QuantizationBehavior.PROPAGATION,
        dtypes=GRID,
        annotator="sima_conv_hardtanh",
        constraints=("currently covered after Conv and Add",),
        cases=("conv_hardtanh", "conv_bn_hardtanh", "add_hardtanh"),
        negative=("standalone clamp and arbitrary min/max coverage is required",),
    ),
    _entry("sigmoid", "Sigmoid", 13, modules=("torch.nn.Sigmoid",), functions=("torch.sigmoid",), aten=("aten.sigmoid.default",), annotator="sima_sigmoid", cases=("sigmoid",), onnx_cases=("sigmoid",)),
    _entry(
        "silu_decomposition",
        "Sigmoid+Mul",
        13,
        modules=("torch.nn.SiLU",),
        functions=("torch.nn.functional.silu",),
        aten=("aten.silu.default", "aten.silu_.default"),
        behavior=QuantizationBehavior.DECOMPOSED,
        annotator="sima_silu",
        onnx=("Sigmoid", "Mul"),
        cases=("silu",),
    ),
    _entry("erf", "Erf", 13, functions=("torch.erf",), aten=("aten.erf.default",), annotator="sima_erf", cases=("erf", "decomposed_gelu"), onnx_cases=("erf",)),
    _entry(
        "exact_gelu_decomposition",
        "Gelu",
        20,
        modules=("torch.nn.GELU",),
        functions=("torch.nn.functional.gelu",),
        aten=("aten.gelu.default",),
        behavior=QuantizationBehavior.DECOMPOSED,
        constraints=("approximate='none' only", "native Gelu is outside opset 17"),
        annotator="sima_gelu",
        onnx=("Div", "Erf", "Add", "Mul"),
        cases=("exact_gelu",),
        onnx_cases=("exact_gelu",),
        negative=("approximate='tanh' is rejected",),
    ),
    _entry(
        "softmax",
        "Softmax",
        13,
        modules=("torch.nn.Softmax",),
        functions=("torch.softmax", "torch.nn.functional.softmax"),
        aten=("aten.softmax.int", "aten._softmax.default"),
        constraints=("AFE layout conversion must map the logical reduction axis to channel",),
        annotator="sima_softmax",
        cases=("softmax",),
        onnx_cases=("softmax",),
    ),
    _entry("log_softmax", "LogSoftmax", 13, modules=("torch.nn.LogSoftmax",), functions=("torch.log_softmax", "torch.nn.functional.log_softmax"), aten=("aten.log_softmax.int",), constraints=("input and output rank >= 2",), annotator="sima_unary_int8", cases=("log_softmax",)),
    _entry("layer_norm", "LayerNormalization", 17, modules=("torch.nn.LayerNorm",), functions=("torch.nn.functional.layer_norm",), aten=("aten.layer_norm.default",), constraints=("FLOAT16 stash_type is unsupported",), annotator="sima_layer_norm", cases=("layer_norm",), onnx_cases=("layer_norm",)),
    _entry("instance_norm", "InstanceNormalization", 6, modules=("torch.nn.InstanceNorm2d",), functions=("torch.nn.functional.instance_norm",), aten=("aten.instance_norm.default",), constraints=("rank 4 or 5",), annotator="sima_unary_int8", cases=("instance_norm",)),
    _entry(
        "local_response_norm",
        "LRN",
        13,
        status=SupportStatus.DEFERRED,
        modules=("torch.nn.LocalResponseNorm",),
        functions=("torch.nn.functional.local_response_norm",),
        aten=("decomposes to pad/avg_pool3d/pow/mul/add/div",),
        behavior=QuantizationBehavior.DECOMPOSED,
        positive=(),
        negative=("do not claim support until the captured avg_pool3d composite is covered",),
    ),
    _entry(
        "mean_variance_normalization",
        "MeanVarianceNormalization",
        13,
        status=SupportStatus.DEFERRED,
        functions=("composite mean/subtract/square/sqrt/divide",),
        aten=("no stable single ATen form",),
        behavior=QuantizationBehavior.DECOMPOSED,
        constraints=("rank <= 4 for portable MLA support", "variance must be nonzero"),
        positive=(),
        negative=("composite topology and zero-variance behavior require tests",),
    ),
    _entry("average_pool", "AveragePool", 11, modules=("torch.nn.AdaptiveAvgPool2d",), functions=("torch.nn.functional.adaptive_avg_pool2d",), aten=("aten.adaptive_avg_pool2d.default",), constraints=("non-global kernel dimensions < 128",), annotator="adaptive_avg_pool2d", cases=("adaptive_avg_pool2d",), onnx_cases=("adaptive_avg_pool2d",)),
    _entry("global_average_pool", "GlobalAveragePool", 13, modules=("torch.nn.AdaptiveAvgPool2d",), functions=("torch.nn.functional.adaptive_avg_pool2d",), aten=("aten.adaptive_avg_pool2d.default",), constraints=("output size is 1x1",), annotator="adaptive_avg_pool2d", cases=("global_average_pool",)),
    _entry("max_pool", "MaxPool", 12, modules=("torch.nn.MaxPool2d",), functions=("torch.nn.functional.max_pool2d",), aten=("aten.max_pool2d.default",), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("dilation=1", "no indices output", "kernel dimensions < 128 unless global"), annotator="propagate_annotation", cases=("max_pool2d",), onnx_cases=("max_pool2d",)),
    _entry("global_max_pool", "GlobalMaxPool", 13, modules=("torch.nn.AdaptiveMaxPool2d",), functions=("torch.nn.functional.adaptive_max_pool2d",), aten=("aten.adaptive_max_pool2d.default", "operator.getitem(value)"), behavior=QuantizationBehavior.MIXED_OUTPUT, annotator="sima_global_max_pool2d", cases=("global_max_pool",)),
    _entry("reduce_mean", "ReduceMean", 13, functions=("torch.mean",), aten=("aten.mean.dim",), constraints=("compiler-supported spatial axes and extents",), annotator="sima_reduction", cases=("reduce_mean",)),
    _entry("reduce_sum", "ReduceSum", 13, functions=("torch.sum",), aten=("aten.sum.dim_IntList",), constraints=("compiler-supported spatial axes and extents",), annotator="sima_reduction", cases=("reduce_sum",)),
    _entry("reduce_max", "ReduceMax", 13, functions=("torch.amax",), aten=("aten.amax.default",), annotator="sima_reduction", cases=("reduce_max",)),
    _entry("reduce_l1", "ReduceL1", 17, functions=("torch.linalg.vector_norm(ord=1)",), aten=("aten.linalg_vector_norm.default",), constraints=("axes are static", "partial spatial extent < 128"), annotator="sima_reduction", cases=("reduce_l1",)),
    _entry("reduce_log_sum_exp", "ReduceLogSumExp", 17, functions=("torch.logsumexp",), aten=("aten.logsumexp.default",), constraints=("axes are static", "partial spatial extent < 128"), annotator="sima_reduction", cases=("reduce_logsumexp",)),
    _entry("reduce_log_sum", "ReduceLogSum", 17, status=SupportStatus.DEFERRED, functions=("torch.log(torch.sum(...))",), aten=("aten.sum.dim_IntList", "aten.log.default"), behavior=QuantizationBehavior.DECOMPOSED, constraints=("positive reduced sums",), positive=(), negative=("add composite lifecycle coverage",)),
    _entry("reduce_sum_square", "ReduceSumSquare", 17, status=SupportStatus.DEFERRED, functions=("torch.sum(torch.square(...))",), aten=("aten.mul.Tensor or aten.pow.Tensor_Scalar", "aten.sum.dim_IntList"), behavior=QuantizationBehavior.DECOMPOSED, positive=(), negative=("add composite lifecycle coverage",)),
    _entry("variadic_mean", "Mean", 13, status=SupportStatus.DEFERRED, functions=("stack/add followed by divide",), aten=("no direct PyTorch ATen equivalent",), behavior=QuantizationBehavior.DECOMPOSED, constraints=("same-shaped inputs", "fewer than 128 inputs"), positive=(), negative=("add an exported composite contract",)),
    _entry("variadic_sum", "Sum", 13, status=SupportStatus.DEFERRED, functions=("sum of a tensor sequence",), aten=("chain of aten.add.Tensor",), behavior=QuantizationBehavior.DECOMPOSED, constraints=("same rank and batch", "broadcast only non-batch dimensions"), positive=(), negative=("add a variadic-input topology test",)),
    _entry("concat", "Concat", 13, functions=("torch.cat",), aten=("aten.cat.default",), constraints=("not along batch axis", "all inputs share a realizable grid"), annotator="sima_cat", cases=("cat",), onnx_cases=("cat",)),
    _entry("expand", "Expand", 13, functions=("torch.Tensor.expand",), aten=("aten.expand.default",), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("static broadcast-compatible shape", "batch cannot change"), annotator="sima_grid_preserving", cases=("expand",)),
    _entry("reshape", "Reshape", 14, functions=("torch.reshape", "torch.Tensor.reshape", "torch.Tensor.view"), aten=("aten.reshape.default", "aten.view.default"), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("allowzero=0", "non-empty shape"), annotator="sima_grid_preserving", cases=("reshape",)),
    _entry("flatten", "Flatten", 21, functions=("torch.flatten",), aten=("aten.flatten.using_ints",), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("batch dimension is preserved", "opset-17 uses the older Flatten schema"), annotator="sima_grid_preserving", cases=("flatten",), negative=("PR 111 records schema 21; keep opset-17 export coverage",)),
    _entry("transpose", "Transpose", 13, functions=("torch.transpose", "torch.permute"), aten=("aten.transpose.int", "aten.permute.default"), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("batch axis is not moved"), annotator="sima_grid_preserving", cases=("transpose",)),
    _entry("depth_to_space", "DepthToSpace", 13, modules=("torch.nn.PixelShuffle",), functions=("torch.pixel_shuffle", "torch.nn.functional.pixel_shuffle"), aten=("aten.pixel_shuffle.default",), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, annotator="sima_grid_preserving", cases=("depth_to_space",)),
    _entry("space_to_depth", "SpaceToDepth", 13, modules=("torch.nn.PixelUnshuffle",), functions=("torch.pixel_unshuffle", "torch.nn.functional.pixel_unshuffle"), aten=("aten.pixel_unshuffle.default",), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, annotator="sima_grid_preserving", cases=("space_to_depth",)),
    _entry("padding", "Pad", 13, functions=("torch.nn.functional.pad",), aten=("aten.pad.default",), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("constant zero mode", "at most two dimensions padded"), annotator="sima_grid_preserving", cases=("pad",), negative=("reject nonzero and non-constant padding",)),
    _entry("slice", "Slice", 13, status=SupportStatus.PARTIAL, functions=("Python tensor slicing", "torch.select", "torch.unsqueeze"), aten=("aten.slice.Tensor", "aten.select.int", "aten.unsqueeze.default"), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("positive steps"), annotator="sima_slice_select_unsqueeze", cases=("slice_select_unsqueeze",), negative=("general slices and negative strides require coverage",)),
    _entry("split", "Split", 13, functions=("torch.split",), aten=("aten.split.Tensor", "aten.split_with_sizes.default", "operator.getitem"), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, annotator="sima_split", cases=("split",)),
    _entry("tile", "Tile", 13, functions=("torch.tile", "torch.Tensor.repeat"), aten=("aten.tile.default", "aten.repeat.default"), behavior=QuantizationBehavior.PROPAGATION, dtypes=GRID, constraints=("static repeats", "batch repeat is 1", "rank >= 2"), annotator="sima_grid_preserving", cases=("tile",)),
    _entry("resize", "Resize", 13, functions=("torch.nn.functional.interpolate",), aten=("aten.upsample_nearest2d.vec", "aten.upsample_bilinear2d.vec"), constraints=("nearest or linear", "not tf_crop_and_resize", "single tensor input"), annotator="sima_unary_int8", cases=("resize_nearest", "resize_bilinear"), negative=("reject unsupported modes and nearest scaling contracts",)),
    _entry("argmax", "ArgMax", 13, functions=("torch.argmax",), aten=("aten.argmax.default",), behavior=QuantizationBehavior.TYPE_PRESERVING, dtypes=("signed INT8 input", "torch.int64 index output; compiler lowers to int32"), constraints=("compiler channel-axis reduction"), annotator="sima_mixed_output", cases=("argmax",), negative=("index output must never receive fake quantization",)),
    _entry("topk", "TopK", 11, functions=("torch.topk",), aten=("aten.topk.default", "operator.getitem(value/index)"), behavior=QuantizationBehavior.MIXED_OUTPUT, dtypes=("signed INT8 values", "torch.int64 indices; compiler lowers to int32"), constraints=("constant K", "1 <= K < axis size", "largest=True", "rank >= 2"), annotator="sima_mixed_output", cases=("topk_values",), negative=("indices remain unquantized", "reject dynamic K and largest=False")),
    _entry(
        "batch_norm_training",
        "BatchNormalization",
        15,
        compiler_int8=False,
        modules=("torch.nn.BatchNorm1d", "torch.nn.BatchNorm2d"),
        functions=("torch.nn.functional.batch_norm",),
        aten=("aten.batch_norm.default", "aten._native_batch_norm_legit.default", "operator.getitem"),
        behavior=QuantizationBehavior.TRAINING_ONLY,
        constraints=("training statistics are frozen and folded before deployment",),
        annotator="sima_batchnorm",
        cases=("batchnorm", "conv_bn", "conv_bn_relu", "conv_bn_hardtanh"),
        negative=("standalone deployment BatchNormalization is not an INT8 compiler kernel",),
    ),
    _entry("embedding_gather", "Gather", 13, status=SupportStatus.REJECTED, compiler_int8=False, modules=("torch.nn.Embedding",), functions=("torch.nn.functional.embedding",), aten=("aten.embedding.default",), behavior=QuantizationBehavior.UNSUPPORTED, dtypes=("integer indices must remain integer",), constraints=("compiler INT8 Gather is unsupported"), positive=(), negative=("preparation rejects embedding instead of quantizing indices",)),
    _entry("grid_sample", "GridSample", 16, status=SupportStatus.REJECTED, compiler_int8=False, functions=("torch.nn.functional.grid_sample",), aten=("aten.grid_sampler.default", "aten.grid_sampler_2d.default"), behavior=QuantizationBehavior.UNSUPPORTED, constraints=("compiler supports BF16 only"), positive=(), negative=("preparation rejects W8A8 GridSample",)),
    _entry("reduce_min", "ReduceMin", 13, status=SupportStatus.REJECTED, compiler_int8=False, functions=("torch.amin",), aten=("aten.amin.default",), behavior=QuantizationBehavior.UNSUPPORTED, constraints=("not currently supported by the compiler"), positive=(), negative=("preparation rejects ReduceMin",)),
    _entry("cumulative_sum", "CumSum", 14, status=SupportStatus.REJECTED, compiler_int8=False, functions=("torch.cumsum",), aten=("aten.cumsum.default",), behavior=QuantizationBehavior.UNSUPPORTED, constraints=("not in the compiler INT8 contract"), positive=(), negative=("preparation must reject or leave CumSum explicitly outside QAT",)),
)


MANIFEST_BY_FAMILY = {entry.family: entry for entry in OPERATOR_MANIFEST}
MANIFEST_BY_TEST_CASE = {
    case_id: entry for entry in OPERATOR_MANIFEST for case_id in entry.test_case_ids
}
ONNX_TEST_CASE_IDS = frozenset(
    case_id
    for entry in OPERATOR_MANIFEST
    if entry.status in {SupportStatus.SUPPORTED, SupportStatus.PARTIAL}
    for case_id in entry.test_case_ids
)
COMPILER_INT8_ONNX_OPS_OPSET17 = frozenset(
    entry.compiler_operator
    for entry in OPERATOR_MANIFEST
    if entry.compiler_opset is not None
    and entry.compiler_opset <= ONNX_OPSET
    and entry.compiler_int8_supported
    and "+" not in entry.compiler_operator
)


__all__ = [
    "COMPILER_INT8_ONNX_OPS_OPSET17",
    "COMPILER_OPERATOR_SOURCE",
    "MANIFEST_BY_FAMILY",
    "MANIFEST_BY_TEST_CASE",
    "ONNX_OPSET",
    "ONNX_TEST_CASE_IDS",
    "OPERATOR_MANIFEST",
    "OPERATOR_MANIFEST_VERSION",
    "OperatorManifestEntry",
    "QuantizationBehavior",
    "SupportStatus",
]
