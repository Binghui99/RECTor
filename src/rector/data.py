"""Trace parsing and windowed feature construction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def read_trace(path: Path) -> list[tuple[float, float]]:
    """Read a tab-separated ``timestamp, packet_size`` trace."""
    packets: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected timestamp<TAB>packet_size")
            packets.append((float(fields[0]), float(fields[1])))
    return packets


def slice_trace(
    packets: Iterable[tuple[float, float]], start: float, end: float, size_threshold: float
) -> list[dict[str, float]]:
    """Extract one window and merge same-timestamp packets into super-packets."""
    features: list[dict[str, float]] = []
    previous_time: float | None = None
    for timestamp, size in packets:
        if timestamp < start:
            continue
        if timestamp > end:
            break
        if abs(size) <= size_threshold:
            continue
        direction = 1.0 if size > 0 else -1.0
        iat = 0.0 if previous_time is None else direction * (timestamp - previous_time)
        if features and timestamp == previous_time:
            features[-1]["size"] += size
        else:
            features.append({"iat": iat, "size": size})
        previous_time = timestamp
    return features


def build_windows(
    data_dir: Path,
    output_dir: Path,
    names: Iterable[str],
    *,
    window_seconds: float = 5.0,
    stride_seconds: float = 2.0,
    num_windows: int = 11,
    ingress_threshold: float = 512.0,
    egress_threshold: float = 66.0,
) -> list[Path]:
    """Create the pickle files consumed by training and evaluation."""
    names = list(names)
    ingress = {name: read_trace(data_dir / "inflow" / name) for name in names}
    egress = {name: read_trace(data_dir / "outflow" / name) for name in names}
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(num_windows):
        start = index * stride_seconds
        end = start + window_seconds
        in_windows, out_windows, labels = [], [], []
        for name in names:
            in_window = slice_trace(ingress[name], start, end, ingress_threshold)
            out_window = slice_trace(egress[name], start, end, egress_threshold)
            if in_window and out_window:
                in_windows.append(in_window)
                out_windows.append(out_window)
                labels.append(name)
        payload = {"ingress": in_windows, "egress": out_windows, "label": labels}
        path = output_dir / f"{window_seconds:g}_win{index}_addn{stride_seconds:g}_superpkt.pickle"
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        written.append(path)
    return written


def load_window(path: Path, pad_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a processed window as normalized, fixed-length feature arrays."""
    with path.open("rb") as handle:
        traces = pickle.load(handle)

    def encode(flows: list[list[dict[str, float]]]) -> np.ndarray:
        encoded = []
        for flow in flows:
            iat = np.asarray([pair["iat"] * 1000.0 for pair in flow], dtype=np.float32)
            size = np.asarray([pair["size"] / 1000.0 for pair in flow], dtype=np.float32)
            if len(iat):
                iat[0] = 0.0
            vector = np.concatenate((iat, size))[:pad_length]
            encoded.append(np.pad(vector, (0, pad_length - len(vector))))
        return np.asarray(encoded, dtype=np.float32)[..., None]

    labels = np.asarray(traces["label"])
    return encode(traces["ingress"]), encode(traces["egress"]), labels

