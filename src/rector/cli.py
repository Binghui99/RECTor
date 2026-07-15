"""Command-line interface for the reproducible data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rector", description="RECTor experiment utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="convert raw traces into window pickle files")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--file-list", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--window-seconds", type=float, default=5.0)
    prepare.add_argument("--stride-seconds", type=float, default=2.0)
    prepare.add_argument("--num-windows", type=int, default=11)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        names = [line.strip() for line in args.file_list.read_text().splitlines() if line.strip()]
        paths = build_windows(
            args.data_dir,
            args.output_dir,
            names,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            num_windows=args.num_windows,
        )
        print(f"Wrote {len(paths)} windows to {args.output_dir}")


if __name__ == "__main__":
    main()

