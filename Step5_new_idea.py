import os
import pickle
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm


# -----------------------
# Data Processing Functions
# -----------------------
def load_window_data(pickle_path):
    """Load one pickle file and extract concatenated iat and size features."""
    with open(pickle_path, 'rb') as handle:
        traces = pickle.load(handle)
    ingress_seq = traces["ingress"]
    egress_seq = traces["egress"]
    labels = traces["label"]

    window_ingress_size = [np.array([float(pair["size"]) / 1000.0 for pair in seq]) for seq in ingress_seq]
    window_egress_size = [np.array([float(pair["size"]) / 1000.0 for pair in seq]) for seq in egress_seq]
    window_ingress_iat = [np.array([float(pair["iat"]) * 1000.0 for pair in seq]) for seq in ingress_seq]
    window_egress_iat = [np.array([float(pair["iat"]) * 1000.0 for pair in seq]) for seq in egress_seq]

    new_window_ingress_iat = [np.concatenate(([0.0], trace[1:])) for trace in window_ingress_iat]
    new_window_egress_iat = [np.concatenate(([0.0], trace[1:])) for trace in window_egress_iat]

    window_ingress = [np.concatenate((iat, size), axis=None) for iat, size in
                      zip(new_window_ingress_iat, window_ingress_size)]
    window_egress = [np.concatenate((iat, size), axis=None) for iat, size in
                     zip(new_window_egress_iat, window_egress_size)]
    circuit_labels = np.array([int(l.split('_')[0]) for l in labels])
    return window_ingress, window_egress, circuit_labels


def pad_windows(window_list, pad_length):
    """Pad or truncate each 1D window to a fixed length and reshape to (length, 1)."""
    padded = []
    for x in window_list:
        x_trunc = x[:pad_length]
        if len(x_trunc) < pad_length:
            x_trunc = np.pad(x_trunc, (0, pad_length - len(x_trunc)), 'constant')
        padded.append(x_trunc.reshape(-1, 1))
    return np.array(padded)


def load_and_stack_test_data(args):
    """
    Loads 11 pickle files from the test directory, pads each window,
    and stacks them so each test sample becomes a flow of shape:
         (num_samples, num_window, window_length, 1)
    """
    windows1_list, windows2_list = [], []
    labels = None

    for win in range(args.num_window):
        pickle_filename = f"{args.win_interval}_win{win}_addn{args.addn}_superpkt.pickle"
        pickle_path = os.path.join(args.test_dir, pickle_filename)
        print(f"Loading test file: {pickle_path}")
        window_ingress, window_egress, circuit_labels = load_window_data(pickle_path)

        pad_t = args.tor_len * 2  # Use same padding as training.
        pad_e = args.exit_len * 2
        padded_ingress = pad_windows(window_ingress, pad_t)
        padded_egress = pad_windows(window_egress, pad_e)

        windows1_list.append(padded_ingress)
        windows2_list.append(padded_egress)
        if labels is None:
            labels = circuit_labels

    test_ingress = np.stack(windows1_list, axis=1)  # (N, num_window, pad_t, 1)
    test_egress = np.stack(windows2_list, axis=1)  # (N, num_window, pad_e, 1)
    print("Test Data Shapes:")
    print("Ingress:", test_ingress.shape, "Egress:", test_egress.shape)
    return test_ingress, test_egress, labels


# -----------------------
# Model Definitions: GRU MIL Siamese
# -----------------------
class GRUWindowEncoder(nn.Module):
    """
    Encodes a single window using a GRU.
    Input: (batch, window_length, 1)
    Output: (batch, hidden_size) or (batch, 2*hidden_size) if bidirectional.
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=1, bidirectional=False):
        super(GRUWindowEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        x = x.contiguous()  # Ensure tensor is contiguous.
        out, h_n = self.gru(x)
        if self.bidirectional:
            forward_h = h_n[-2, :, :]
            backward_h = h_n[-1, :, :]
            h = torch.cat([forward_h, backward_h], dim=1)
        else:
            h = h_n[-1, :, :]
        return h


class AttentionAggregator(nn.Module):
    """
    Aggregates window embeddings using an attention mechanism.
    Input: (batch, num_windows, emb_size)
    Output: (batch, emb_size)
    """

    def __init__(self, emb_size):
        super(AttentionAggregator, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, 1)
        )

    def forward(self, H):
        attn_scores = self.attention(H)  # (batch, num_windows, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        aggregated = torch.sum(attn_weights * H, dim=1)  # (batch, emb_size)
        return aggregated, attn_weights


class GRU_MIL_Siamese(nn.Module):
    """
    Overall feature extractor that processes a flow with 11 windows.
    Input: (batch, num_windows, window_length, 1)
    Output: final flow embedding and attention weights.
    """

    def __init__(self, input_size=1, window_length=1000, num_windows=11,
                 hidden_size=64, gru_layers=1, bidirectional=False):
        super(GRU_MIL_Siamese, self).__init__()
        self.num_windows = num_windows
        self.window_encoder = GRUWindowEncoder(input_size, hidden_size, gru_layers, bidirectional)
        final_hidden_size = hidden_size * (2 if bidirectional else 1)
        self.attention_aggregator = AttentionAggregator(final_hidden_size)
        self.fc = nn.Linear(final_hidden_size, final_hidden_size)

    def forward(self, x):
        # x: (batch, num_windows, window_length, 1)
        batch_size, num_windows, window_length, _ = x.size()
        x_reshaped = x.view(batch_size * num_windows, window_length, 1)
        x_reshaped = x_reshaped.contiguous()  # Make sure tensor is contiguous.
        window_embeddings = self.window_encoder(x_reshaped)
        emb_size = window_embeddings.size(-1)
        window_embeddings = window_embeddings.view(batch_size, num_windows, emb_size)
        aggregated, attn_weights = self.attention_aggregator(window_embeddings)
        final_emb = self.fc(aggregated)
        return final_emb, attn_weights


# -----------------------
# Utility Functions for Mapping & Evaluation
# -----------------------
def compute_distance_matrix(emb_ingress, emb_egress):
    """
    Compute an N x N cosine distance matrix.
    Distance = 1 - cosine_similarity.
    """
    emb_i_norm = emb_ingress / emb_ingress.norm(dim=1, keepdim=True)
    emb_e_norm = emb_egress / emb_egress.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(emb_i_norm, emb_e_norm.t())
    distance_matrix = 1 - similarity_matrix
    return distance_matrix


def map_1_to_1(distance_matrix):
    """
    Use the Hungarian algorithm to find the optimal 1-to-1 assignment.
    Returns a list of tuples (ingress_index, egress_index, distance).
    """
    distance_np = distance_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(distance_np)
    mapping = [(i, j, distance_np[i, j]) for i, j in zip(row_ind, col_ind)]
    return mapping


def map_partial(distance_matrix, threshold):
    """
    For each ingress flow, select egress flows with distance < threshold.
    Returns a dictionary mapping ingress index to a list of egress indices.
    """
    distance_np = distance_matrix.cpu().numpy()
    mapping = {}
    N = distance_np.shape[0]
    for i in range(N):
        matched = np.where(distance_np[i, :] < threshold)[0]
        mapping[i] = matched.tolist()
    return mapping


def evaluate_thresholds(distance_matrix, thresholds):
    """
    Evaluate TPR and FPR over a range of thresholds.
    Assumes ground truth is that the correct match is on the diagonal.
    """
    distance_np = distance_matrix.cpu().numpy()
    N = distance_np.shape[0]
    positives = np.diag(distance_np)

    mask = np.ones_like(distance_np, dtype=bool)
    np.fill_diagonal(mask, 0)
    negatives = distance_np[mask]

    tpr_list = []
    fpr_list = []
    for t in thresholds:
        tp = np.sum(positives < t)
        fp = np.sum(negatives < t)
        tpr = tp / N
        fpr = fp / (N * (N - 1))
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


# -----------------------
# Main Testing & Mapping Pipeline
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tor_len', default=500, type=int)
    parser.add_argument('--exit_len', default=500, type=int)
    parser.add_argument('--win_interval', default=5, type=int)
    parser.add_argument('--num_window', default=11, type=int)
    parser.add_argument('--addn', default=2, type=int)
    parser.add_argument('--test_dir', default=".\\Processed-test-set\\",
                        help="Directory containing test pickle files")
    parser.add_argument('--model_path', default="crawle_overlap_new2021__11_GRU_MIL_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument('--output_dir', default="test_results_new",
                        help="Directory to save test outputs")
    parser.add_argument('--mapping_type', default="one2one", choices=["one2one", "partial"],
                        help="Mapping scenario: 'one2one' for full pairing or 'partial' for sparse matching")
    parser.add_argument('--threshold', default=0.5, type=float,
                        help="Threshold for partial mapping (cosine distance)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # If you still see cuDNN errors, try disabling cuDNN:
    # torch.backends.cudnn.enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Process test data.
    test_ingress, test_egress, _ = load_and_stack_test_data(args)
    test_ingress_tensor = torch.FloatTensor(test_ingress).to(device)
    test_egress_tensor = torch.FloatTensor(test_egress).to(device)

    # Load the trained feature extractor.
    pad_length = args.tor_len * 2  # Assumed equal to exit length.
    model = GRU_MIL_Siamese(input_size=1, window_length=pad_length, num_windows=args.num_window,
                            hidden_size=64, gru_layers=1, bidirectional=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Extract embeddings.
    with torch.no_grad():
        emb_ingress, _ = model(test_ingress_tensor)  # Shape: (N, d)
        emb_egress, _ = model(test_egress_tensor)  # Shape: (N, d)

    # Compute full distance matrix.
    distance_matrix = compute_distance_matrix(emb_ingress, emb_egress)
    distance_matrix_np = distance_matrix.cpu().numpy()

    # Save the N x N distance matrix.
    distance_matrix_path = os.path.join(args.output_dir, "distance_matrix.npy")
    np.save(distance_matrix_path, distance_matrix_np)
    print(f"Distance matrix saved to {distance_matrix_path}")

    # Evaluate thresholds for ROC.
    thresholds = np.linspace(distance_matrix_np.min(), distance_matrix_np.max(), num=100)
    tpr_list, fpr_list = evaluate_thresholds(distance_matrix, thresholds)
    roc_data = np.column_stack((thresholds, tpr_list, fpr_list))
    roc_data_path = os.path.join(args.output_dir, "tpr_fpr.csv")
    np.savetxt(roc_data_path, roc_data, delimiter=",", header="threshold,tpr,fpr", comments='')
    print(f"TPR vs FPR data saved to {roc_data_path}")

    # Mapping evaluation.
    if args.mapping_type == "one2one":
        mapping = map_1_to_1(distance_matrix)
        mapping_csv_path = os.path.join(args.output_dir, "mapping_one2one.csv")
        with open(mapping_csv_path, "w") as f:
            f.write("ingress_index,egress_index,distance\n")
            for i, j, d in mapping:
                f.write(f"{i},{j},{d:.6f}\n")
        print(f"1-to-1 mapping results saved to {mapping_csv_path}")
    else:
        mapping = map_partial(distance_matrix, args.threshold)
        mapping_csv_path = os.path.join(args.output_dir, "mapping_partial.csv")
        with open(mapping_csv_path, "w") as f:
            f.write("ingress_index,matched_egress_indices\n")
            for i, matches in mapping.items():
                f.write(f"{i},{matches}\n")
        print(f"Partial mapping results (threshold = {args.threshold}) saved to {mapping_csv_path}")


if __name__ == "__main__":
    main()
