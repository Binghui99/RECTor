import numpy as np

from rector.matching import cosine_distance_matrix, one_to_one_mapping, threshold_metrics


def test_matching_recovers_diagonal_pairs():
    embeddings = np.eye(3)
    distances = cosine_distance_matrix(embeddings, embeddings)
    assert [pair[:2] for pair in one_to_one_mapping(distances)] == [(0, 0), (1, 1), (2, 2)]
    metrics = threshold_metrics(distances, np.asarray([0.1]))
    np.testing.assert_allclose(metrics[0], [0.1, 1.0, 0.0])

