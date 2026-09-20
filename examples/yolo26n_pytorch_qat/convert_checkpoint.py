#!/usr/bin/env python3
"""One-time conversion of an Ultralytics YOLO26n pickle to a pure state dict.

This is the only file in the workflow that imports Ultralytics.  Neither the
training process nor its checkpoints depend on that package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from model import build_yolo26n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Ultralytics yolo26n.pt checkpoint")
    parser.add_argument("--output", required=True, help="Portable state-dict output")
    parser.add_argument(
        "--ultralytics-source",
        help="Optional checkout containing the ultralytics Python package",
    )
    parser.add_argument("--verify-size", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ultralytics_source:
        sys.path.insert(0, str(Path(args.ultralytics_source).resolve()))
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Checkpoint conversion requires Ultralytics once. Install it or pass "
            "--ultralytics-source; QAT training itself does not import it."
        ) from error

    source = YOLO(args.source).model.float().cpu().eval()
    portable = build_yolo26n().eval()
    portable.load_state_dict(source.state_dict(), strict=True)

    # Ask the vendor head for raw training outputs without switching its child
    # BatchNorm layers back to training mode.
    source.model[-1].training = True
    generator = torch.Generator().manual_seed(0)
    example = torch.randn(
        1,
        3,
        args.verify_size,
        args.verify_size,
        generator=generator,
    )
    with torch.no_grad():
        source_outputs = source(example)
        portable_outputs = portable(example)
    for branch in ("one2many", "one2one"):
        for name in ("boxes", "scores"):
            torch.testing.assert_close(
                portable_outputs[branch][name],
                source_outputs[branch][name],
                rtol=0,
                atol=0,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sima-pure-pytorch-yolo26n-v1",
            "state_dict": portable.state_dict(),
            "metadata": {
                "classes": 80,
                "strides": (8, 16, 32),
                "reg_max": 1,
                "parameters": 2_572_280,
            },
        },
        output,
    )
    print(f"Wrote portable YOLO26n state dict: {output}")


if __name__ == "__main__":
    main()
