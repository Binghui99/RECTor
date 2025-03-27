import pickle
import numpy as np
import os
import random
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from fontTools.misc.cython import returns
import torch.optim as optim
from Model_for_FNE import DFModel, TransformerEncoderFeatureExtractor
# -----------------------
# Argument Parsing
# -----------------------
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tor_len', default=500, type=int)
    parser.add_argument('--exit_len', default=500, type=int)
    parser.add_argument('--win_interval', default=5, type=int)
    parser.add_argument('--num_window', default=11, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--addn', default=2, type=int)
    parser.add_argument('--input',
                        default='.\\DeepCoffea-data\\')
    parser.add_argument('--test', default='')
    parser.add_argument('--model', default="crawle_overlap_new2021_")
    args = parser.parse_args()
    return args


# pickle_path = '.\\Processed-window-files\\5_win0_addn2_superpkt.pickle'


# -----------------------
# Dataset for Triplet Training
# -----------------------
class TripletDataset(Dataset):
    def __init__(self, anchor_data, positive_data):
        """
        anchor_data: numpy array of shape (N, L1, 1)
        positive_data: numpy array of shape (N, L2, 1)

        Here, we assume that the i-th anchor corresponds to the i-th positive.
        For negative examples, we randomly sample another index from positive_data.
        """
        self.anchor_data = anchor_data
        self.positive_data = positive_data

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        anchor = self.anchor_data[idx]
        positive = self.positive_data[idx]
        # Sample a negative index different from idx
        neg_idx = random.choice([i for i in range(len(self.anchor_data)) if i != idx])
        negative = self.positive_data[neg_idx]
        return (
            torch.FloatTensor(anchor).transpose(0, 1),  # transpose to (channels, length)
            torch.FloatTensor(positive).transpose(0, 1),
            torch.FloatTensor(negative).transpose(0, 1)
        )


# -----------------------
# Cosine Triplet Loss
# -----------------------
def cosine_triplet_loss(anchor, positive, negative, alpha):
    # Compute cosine similarities along channel dimension (assumes embeddings are 1D vectors)
    pos_sim = F.cosine_similarity(anchor, positive, dim=1)
    neg_sim = F.cosine_similarity(anchor, negative, dim=1)
    losses = F.relu(neg_sim - pos_sim + alpha)
    return losses.mean()


# -----------------------
# Data Loading Functions (simplified)
# -----------------------
"""Load one pickle file and extract concatenated iat and size features.
   Returns:
     ingress_windows: list of numpy arrays (each 1D)
     exit_windows: list of numpy arrays (each 1D)
     circuit_labels: numpy array of integers (from file label)
"""
def load_window_data(pickle_path):
    with open(pickle_path, 'rb') as handle:
        traces = pickle.load(handle)
    ingress_seq = traces["ingress"]
    egress_seq = traces["egress"]
    labels = traces["label"]

    # Extract size and iat features (normalize as in original code)
    window_ingress_size = [np.array([float(pair["size"]) / 1000.0 for pair in seq]) for seq in ingress_seq]
    window_egress_size = [np.array([float(pair["size"]) / 1000.0 for pair in seq]) for seq in egress_seq]
    window_ingress_iat = [np.array([float(pair["iat"]) * 1000.0 for pair in seq]) for seq in ingress_seq]
    window_egress_iat = [np.array([float(pair["iat"]) * 1000.0 for pair in seq]) for seq in egress_seq]

    # Set the first iat to 0 for each trace
    new_window_ingress_iat = [np.concatenate(([0.0], trace[1:])) for trace in window_ingress_iat]
    new_window_egress_iat = [np.concatenate(([0.0], trace[1:])) for trace in window_egress_iat]

    # Concatenate iat and size features
    window_ingress = [np.concatenate((iat, size), axis=None) for iat, size in zip(new_window_ingress_iat, window_ingress_size)]
    window_egress = [np.concatenate((iat, size), axis=None) for iat, size in zip(new_window_egress_iat, window_egress_size)]

    # For simplicity, we ignore splitting by circuit labels here.
    circuit_labels = np.array([int(l.split('_')[0]) for l in labels])
    # print(circuit_labels)
    return window_ingress, window_egress, circuit_labels


# window_ingress, window_egress, circuit_labels = load_window_data(pickle_path)

def pad_windows(window_list, pad_length):
    """Pad or truncate each 1D window to a fixed length and reshape to (length,1)."""
    padded = []
    for x in window_list:
        # if len(x) < 100:
            # print('---------- There is something wrong during processing')
            # print(len(x))
        # truncate if necessary
        x_trunc = x[:pad_length]
        # pad with zeros if needed
        if len(x_trunc) < pad_length:
            x_trunc = np.pad(x_trunc, (0, pad_length - len(x_trunc)), 'constant')
        padded.append(x_trunc.reshape(-1, 1))
    return np.array(padded)


def load_and_combine_data(args):
    """ Reads 11 pickle files, processes, and combines them into a single dataset. """
    prefix_pickle_output = ".\\Processed-window-files\\"

    train_windows1, train_windows2, train_labels = [], [], []

    for win in range(11):  # Loop through 11 window files
        pickle_filename = f"{args.win_interval}_win{win}_addn{args.addn}_superpkt.pickle"
        pickle_path = os.path.join(prefix_pickle_output, pickle_filename)

        print(f"Loading: {pickle_path}")
        window_ingress, window_egress, circuit_labels = load_window_data(pickle_path)

        # Define padding sizes
        pad_t = args.tor_len * 2
        pad_e = args.exit_len * 2

        # Apply padding
        padded_ingress = pad_windows(window_ingress, pad_t)
        padded_egress = pad_windows(window_egress, pad_e)

        print(f"After padding, tor shape: {padded_ingress.shape}, exit shape: {padded_egress.shape}")

        # Append to training set
        train_windows1.append(padded_ingress)
        train_windows2.append(padded_egress)
        train_labels.append(circuit_labels)

    # Convert lists to NumPy arrays
    train_windows1 = np.concatenate(train_windows1, axis=0)
    train_windows2 = np.concatenate(train_windows2, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    print("Final Training Data Shapes:")
    print("Train Windows 1 (Tor):", train_windows1.shape)
    print("Train Windows 2 (Exit):", train_windows2.shape)
    print("Train Labels:", train_labels.shape)

    return train_windows1, train_windows2, train_labels






# -----------------------
# Main Training Pipeline
# -----------------------
def main():
    args = get_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device:', device)

    # For simplicity, we load only one window (e.g. window_index=0)
    addn = 3
    # prefix_pickle_output = ".\\Processed-window-files\\"
    # pickle_filename = f"{prefix_pickle_output}{args.win_interval}_win{win}_addn{addn}_superpkt.pickle"
    # pickle_path = os.path.join(args.input, pickle_filename)
    # window_ingress, window_egress, circuit_labels = load_window_data(pickle_path)

    # In original code pad_t = tor_len * 2, pad_e = exit_len * 2
    pad_t = args.tor_len * 2
    pad_e = args.exit_len * 2
    #
    # print("Before padding, #tor windows: ", len(window_ingress))
    # # Pad/truncate windows (using tor_windows for anchor and exit_windows for positive)
    # train_windows1 = pad_windows(window_ingress, pad_t)
    # train_windows2 = pad_windows(window_egress, pad_e)
    train_windows1, train_windows2, train_labels = load_and_combine_data(args)

    print("After padding, tor shape:", train_windows1.shape, "exit shape:", train_windows2.shape)

    # Create dataset and dataloader. Note: we assume a one-to-one correspondence between train_windows1 and train_windows2.
    dataset = TripletDataset(train_windows1, train_windows2)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Define input shapes for the two models.
    # PyTorch Conv1d expects input shape (batch, channels, length)
    input_shape1 = (pad_t, 1)  # for tor
    input_shape2 = (pad_e, 1)  # for exit
    # input_shape1 = (1, 1, pad_t)  # Ensure (batch_size, channels, sequence_length)
    # input_shape2 = (1, 1, pad_e)  # for exit
    emb_size = 128
    # if pad_e == pad_t:
    # Create models. (They are not “shared” in PyTorch unless you explicitly tie their weights.)
    # model_common = DFModel(input_shape1, emb_size, model_name='common').to(device)
    # model_exit = DFModel(input_shape2, emb_size, model_name='exit').to(device)
    model_common = TransformerEncoderFeatureExtractor(input_shape1, emb_size).to(device)

    # Optimizer (using SGD with parameters similar to the original)
    optimizer = optim.SGD(list(model_common.parameters()),
                          lr=0.001, momentum=0.9, weight_decay=1e-6, nesterov=True)
    alpha_value = args.alpha
    nb_epochs = 1000  # set to desired number of epochs (original code used 10000 epochs)
    best_loss = float('inf')

    for epoch in range(nb_epochs):
        model_common.train()
        epoch_loss = 0.0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            # Move data to device
            anchor = anchor.to(device)  # shape: (batch, channels, length)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            # Forward pass through each network
            emb_anchor = model_common(anchor)  # (batch, emb_size)
            emb_positive = model_common(positive)  # (batch, emb_size)
            emb_negative = model_common(negative)  # (batch, emb_size)

            loss = cosine_triplet_loss(emb_anchor, emb_positive, emb_negative, alpha_value)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{nb_epochs} - Loss: {avg_loss:.6f}")

        # Save models if loss improved.
        if avg_loss < best_loss:
            print(f"Loss improved from {best_loss:.6f} to {avg_loss:.6f}. Saving models.")
            best_loss = avg_loss
            torch.save(model_common.state_dict(), args.model + f"{args.num_window}_FNE_Transformer_model.pth")
            # torch.save(model_exit.state_dict(), args.model + f"{args.num_window}_exit_model.pth")
    # # (Optional) Save final test windows
    # test_save_path = os.path.join(args.test,
    #                               f"{args.win_interval}_test_{args.num_window}_addn{addn}_w_superpkt.npz")
    # np.savez_compressed(test_save_path, tor=train_windows1, exit=train_windows2)
if __name__ == "__main__":
    main()


