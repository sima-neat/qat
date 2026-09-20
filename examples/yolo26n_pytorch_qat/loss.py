"""Standalone YOLO26 detection assignment and loss for the raw dual head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes.chunk(2, dim=-1)
    half = size / 2
    return torch.cat((center - half, center + half), dim=-1)


def xyxy_to_xywh(boxes: Tensor) -> Tensor:
    top_left, bottom_right = boxes.chunk(2, dim=-1)
    return torch.cat(((top_left + bottom_right) / 2, bottom_right - top_left), dim=-1)


def distances_to_boxes(distances: Tensor, anchors: Tensor) -> Tensor:
    left_top, right_bottom = distances.chunk(2, dim=-1)
    return torch.cat((anchors - left_top, anchors + right_bottom), dim=-1)


def boxes_to_distances(anchors: Tensor, boxes: Tensor) -> Tensor:
    top_left, bottom_right = boxes.chunk(2, dim=-1)
    return torch.cat((anchors - top_left, bottom_right - anchors), dim=-1)


def complete_iou(first: Tensor, second: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Elementwise complete IoU with broadcastable ``[..., 4]`` xyxy inputs."""

    first_x1, first_y1, first_x2, first_y2 = first.chunk(4, dim=-1)
    second_x1, second_y1, second_x2, second_y2 = second.chunk(4, dim=-1)
    first_w = first_x2 - first_x1
    first_h = first_y2 - first_y1 + epsilon
    second_w = second_x2 - second_x1
    second_h = second_y2 - second_y1 + epsilon
    intersection = (
        (first_x2.minimum(second_x2) - first_x1.maximum(second_x1)).clamp(min=0)
        * (first_y2.minimum(second_y2) - first_y1.maximum(second_y1)).clamp(min=0)
    )
    union = first_w * first_h + second_w * second_h - intersection + epsilon
    iou = intersection / union
    convex_w = first_x2.maximum(second_x2) - first_x1.minimum(second_x1)
    convex_h = first_y2.maximum(second_y2) - first_y1.minimum(second_y1)
    convex_diagonal = convex_w.square() + convex_h.square() + epsilon
    center_distance = (
        (second_x1 + second_x2 - first_x1 - first_x2).square()
        + (second_y1 + second_y2 - first_y1 - first_y2).square()
    ) / 4
    aspect = (4 / math.pi**2) * (
        (second_w / second_h).atan() - (first_w / first_h).atan()
    ).square()
    with torch.no_grad():
        aspect_weight = aspect / (aspect - iou + 1 + epsilon)
    return iou - center_distance / convex_diagonal - aspect * aspect_weight


def make_anchors(features: list[Tensor], strides: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    points = []
    stride_values = []
    for feature, stride in zip(features, strides):
        height, width = feature.shape[-2:]
        x = torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5
        y = torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points.append(torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        stride_values.append(
            torch.full(
                (height * width, 1),
                stride,
                device=feature.device,
                dtype=feature.dtype,
            )
        )
    return torch.cat(points), torch.cat(stride_values)


class TaskAlignedAssigner(nn.Module):
    """Memory-bounded task-aligned assignment used by both YOLO26 heads."""

    def __init__(
        self,
        classes: int,
        topk: int,
        secondary_topk: int | None,
        strides: tuple[int, ...],
        alpha: float = 0.5,
        beta: float = 6.0,
        epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.topk = topk
        self.secondary_topk = secondary_topk or topk
        self.stride_floor = strides[1] if len(strides) > 1 else strides[0]
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(
        self,
        scores: Tensor,
        boxes: Tensor,
        anchors: Tensor,
        labels: Tensor,
        target_boxes: Tensor,
        valid_targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        assigned_boxes = []
        assigned_scores = []
        foreground = []
        assigned_indices = []
        for batch_index in range(scores.shape[0]):
            valid = valid_targets[batch_index, :, 0].bool()
            result = self._assign_image(
                scores[batch_index],
                boxes[batch_index],
                anchors,
                labels[batch_index, valid, 0].long(),
                target_boxes[batch_index, valid],
            )
            image_boxes, image_scores, image_foreground, image_indices = result
            assigned_boxes.append(image_boxes)
            assigned_scores.append(image_scores)
            foreground.append(image_foreground)
            assigned_indices.append(image_indices)
        return (
            torch.stack(assigned_boxes),
            torch.stack(assigned_scores),
            torch.stack(foreground),
            torch.stack(assigned_indices),
        )

    def _assign_image(
        self,
        scores: Tensor,
        boxes: Tensor,
        anchors: Tensor,
        labels: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        anchor_count = boxes.shape[0]
        if targets.numel() == 0:
            return (
                torch.zeros_like(boxes),
                torch.zeros_like(scores),
                torch.zeros(anchor_count, dtype=torch.bool, device=boxes.device),
                torch.zeros(anchor_count, dtype=torch.long, device=boxes.device),
            )

        candidate_targets = xyxy_to_xywh(targets.clone())
        candidate_targets[..., 2:] = torch.where(
            candidate_targets[..., 2:] < self.stride_floor,
            candidate_targets.new_tensor(float(self.stride_floor)),
            candidate_targets[..., 2:],
        )
        candidate_targets = xywh_to_xyxy(candidate_targets)
        left_top = candidate_targets[:, None, :2]
        right_bottom = candidate_targets[:, None, 2:]
        inside = ((anchors[None] - left_top) > self.epsilon).all(dim=-1)
        inside &= ((right_bottom - anchors[None]) > self.epsilon).all(dim=-1)

        class_probabilities = scores[:, labels].transpose(0, 1)
        overlaps = complete_iou(targets[:, None], boxes[None]).squeeze(-1).clamp(min=0)
        alignment = class_probabilities.pow(self.alpha) * overlaps.pow(self.beta)
        alignment = alignment * inside

        selected = torch.zeros_like(inside)
        candidate_count = min(self.topk, anchor_count)
        top_indices = alignment.topk(candidate_count, dim=-1).indices
        selected.scatter_(1, top_indices, True)
        selected &= inside

        collisions = selected.sum(dim=0) > 1
        if collisions.any():
            best_target = overlaps[:, collisions].argmax(dim=0)
            selected[:, collisions] = False
            selected[best_target, collisions] = True

        if self.secondary_topk != self.topk:
            narrowed = torch.zeros_like(selected)
            narrowed_indices = (alignment * selected).topk(
                min(self.secondary_topk, anchor_count), dim=-1
            ).indices
            narrowed.scatter_(1, narrowed_indices, True)
            selected &= narrowed

        foreground = selected.any(dim=0)
        target_index = selected.to(torch.int64).argmax(dim=0)
        output_boxes = targets[target_index]
        output_scores = torch.zeros_like(scores)
        output_scores.scatter_(1, labels[target_index, None], 1.0)
        output_scores *= foreground[:, None]

        positive_alignment = alignment * selected
        best_alignment = positive_alignment.amax(dim=-1, keepdim=True)
        best_overlap = (overlaps * selected).amax(dim=-1, keepdim=True)
        normalized = positive_alignment * best_overlap / (best_alignment + self.epsilon)
        output_scores *= normalized.amax(dim=0)[:, None]
        return output_boxes, output_scores, foreground, target_index


@dataclass(frozen=True)
class LossGains:
    box: float = 7.5
    classification: float = 0.5
    l1: float = 1.5


class DetectionHeadLoss(nn.Module):
    def __init__(
        self,
        classes: int,
        strides: tuple[int, ...],
        topk: int,
        secondary_topk: int | None,
        gains: LossGains,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.strides = strides
        self.gains = gains
        self.assigner = TaskAlignedAssigner(
            classes,
            topk,
            secondary_topk,
            strides,
        )

    @staticmethod
    def _padded_targets(
        batch: dict[str, Tensor],
        batch_size: int,
        image_size: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_indices = batch["batch_idx"].to(device=device, dtype=torch.long)
        classes = batch["cls"].to(device=device, dtype=dtype).reshape(-1)
        boxes = batch["bboxes"].to(device=device, dtype=dtype).reshape(-1, 4)
        counts = torch.bincount(batch_indices, minlength=batch_size)
        maximum = int(counts.max().item()) if counts.numel() else 0
        labels_out = torch.zeros((batch_size, maximum, 1), device=device, dtype=dtype)
        boxes_out = torch.zeros((batch_size, maximum, 4), device=device, dtype=dtype)
        valid_out = torch.zeros((batch_size, maximum, 1), device=device, dtype=torch.bool)
        scale = image_size[[1, 0, 1, 0]]
        for image_index in range(batch_size):
            selected = batch_indices == image_index
            count = int(selected.sum().item())
            if count == 0:
                continue
            labels_out[image_index, :count, 0] = classes[selected]
            boxes_out[image_index, :count] = xywh_to_xyxy(boxes[selected] * scale)
            valid_out[image_index, :count, 0] = True
        return labels_out, boxes_out, valid_out

    def forward(
        self,
        predictions: dict[str, Tensor | list[Tensor]],
        batch: dict[str, Tensor],
    ) -> Tensor:
        box_distances = predictions["boxes"].permute(0, 2, 1).contiguous()
        class_logits = predictions["scores"].permute(0, 2, 1).contiguous()
        features = predictions["feats"]
        if not isinstance(features, list):
            features = list(features)
        anchors, stride = make_anchors(features, self.strides)
        predicted_boxes = distances_to_boxes(box_distances, anchors)
        image_size = torch.tensor(
            features[0].shape[-2:],
            device=class_logits.device,
            dtype=class_logits.dtype,
        ) * self.strides[0]
        labels, target_boxes, valid_targets = self._padded_targets(
            batch,
            class_logits.shape[0],
            image_size,
            class_logits.dtype,
            class_logits.device,
        )
        assigned_boxes, assigned_scores, foreground, _ = self.assigner(
            class_logits.detach().sigmoid(),
            (predicted_boxes.detach() * stride).to(target_boxes.dtype),
            anchors * stride,
            labels,
            target_boxes,
            valid_targets,
        )
        score_sum = assigned_scores.sum().clamp(min=1)
        classification = F.binary_cross_entropy_with_logits(
            class_logits,
            assigned_scores.to(class_logits.dtype),
            reduction="sum",
        ) / score_sum

        if foreground.any():
            weights = assigned_scores[foreground].sum(dim=-1, keepdim=True)
            target_feature_boxes = assigned_boxes / stride
            iou = complete_iou(
                predicted_boxes[foreground],
                target_feature_boxes[foreground],
            )
            box = ((1 - iou) * weights).sum() / score_sum

            target_distances = boxes_to_distances(anchors, target_feature_boxes)
            target_normalized = target_distances * stride
            predicted_normalized = box_distances * stride
            target_normalized[..., 0::2] /= image_size[1]
            target_normalized[..., 1::2] /= image_size[0]
            predicted_normalized[..., 0::2] /= image_size[1]
            predicted_normalized[..., 1::2] /= image_size[0]
            l1 = (
                F.l1_loss(
                    predicted_normalized[foreground],
                    target_normalized[foreground],
                    reduction="none",
                ).mean(dim=-1, keepdim=True)
                * weights
            ).sum() / score_sum
        else:
            empty_gradient = box_distances[..., :0].sum()
            box = empty_gradient
            l1 = empty_gradient

        return class_logits.new_tensor(
            (self.gains.box, self.gains.classification, self.gains.l1)
        ) * torch.stack((box, classification, l1)) * class_logits.shape[0]


class YOLO26Loss(nn.Module):
    """YOLO26 dual-head objective with its progressive head-weight schedule."""

    def __init__(
        self,
        classes: int = 80,
        strides: tuple[int, ...] = (8, 16, 32),
        gains: LossGains = LossGains(),
    ) -> None:
        super().__init__()
        self.one2many = DetectionHeadLoss(classes, strides, 10, None, gains)
        self.one2one = DetectionHeadLoss(classes, strides, 7, 1, gains)

    @staticmethod
    def head_weights(epoch: int, epochs: int) -> tuple[float, float]:
        progress = min(max(epoch, 0), max(epochs - 1, 0)) / max(epochs - 1, 1)
        one2many = 0.8 + (0.1 - 0.8) * progress
        return one2many, 1.0 - one2many

    def forward(
        self,
        predictions: dict[str, dict[str, Tensor | list[Tensor]]],
        batch: dict[str, Tensor],
        epoch: int,
        epochs: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        one2many_weight, one2one_weight = self.head_weights(epoch, epochs)
        one2many = self.one2many(predictions["one2many"], batch)
        one2one = self.one2one(predictions["one2one"], batch)
        components = one2many * one2many_weight + one2one * one2one_weight
        metrics = {
            "box": components[0].detach(),
            "classification": components[1].detach(),
            "l1": components[2].detach(),
            "one2many_weight": components.new_tensor(one2many_weight),
        }
        return components.sum(), metrics
