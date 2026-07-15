from pathlib import Path

import numpy as np

from rector.data import load_window, slice_trace


def test_slice_trace_filters_and_merges_packets():
    packets = [(0.0, 600.0), (0.0, 700.0), (1.0, -800.0), (6.0, 900.0)]
    assert slice_trace(packets, 0.0, 5.0, 512.0) == [
        {"iat": 0.0, "size": 1300.0},
        {"iat": -1.0, "size": -800.0},
    ]


def test_load_window_normalizes_and_pads(tmp_path: Path):
    import pickle

    path = tmp_path / "window.pickle"
    payload = {
        "ingress": [[{"iat": 1.0, "size": 500.0}]],
        "egress": [[{"iat": 2.0, "size": -1000.0}]],
        "label": ["3_7"],
    }
    path.write_bytes(pickle.dumps(payload))
    ingress, egress, labels = load_window(path, 4)
    assert ingress.shape == egress.shape == (1, 4, 1)
    np.testing.assert_allclose(ingress[0, :, 0], [0.0, 0.5, 0.0, 0.0])
    assert labels.tolist() == ["3_7"]

