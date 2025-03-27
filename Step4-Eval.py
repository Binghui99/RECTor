import torch
import torch.nn as nn
import torch.nn.functional as F
import Model_for_FNE
from Model_for_FNE import DFModel
import argparse
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import csv

parser = argparse.ArgumentParser ()
parser.add_argument ('-test', default=".\\Processd-test-set\\")
parser.add_argument ('-flow', default=2094)
parser.add_argument ('-tor_len', default=500)
parser.add_argument ('-exit_len', default=500)
parser.add_argument ('-model1', default='crawle_overlap_new2021_11_FNE_model.pth')
parser.add_argument ('-model2', default='crawle_overlap_new2021_11_FNE_model.pth')
parser.add_argument ('-output', default="results.csv")
parser.add_argument('--win_interval', default=5, type=int)
parser.add_argument('--num_window', default=11, type=int)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--addn', default=2, type=int)
args = parser.parse_args ()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:', device)


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

def ini_cosine_output(single_output_l, input_number):
    for pairs in range (0, (input_number * input_number)):
        single_output_l.append (0)  # ([0])

def calculate_bdr(tpr, fpr):
    TPR = tpr
    FPR = fpr
    c = 1 / int(args.flow)
    u = (int(args.flow)-1) / int(args.flow)
    if ((TPR * c) + (FPR * u)) != 0:
        BDR = (TPR * c) / ((TPR * c) + (FPR * u))
    else:
        BDR = -1
    return BDR

def load_pickle_data(pickle_path):

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
    window_ingress = [np.concatenate((iat, size), axis=None) for iat, size in
                      zip(new_window_ingress_iat, window_ingress_size)]
    window_egress = [np.concatenate((iat, size), axis=None) for iat, size in
                     zip(new_window_egress_iat, window_egress_size)]

    # For simplicity, we ignore splitting by circuit labels here.
    circuit_labels = np.array([int(l.split('_')[0]) for l in labels])
    # print(circuit_labels)
    return window_ingress, window_egress, circuit_labels

def load_and_combine_data(args):
    """ Reads 11 pickle files, processes, and combines them into a single dataset. """
    prefix_pickle_output = ".\\Processed-test-set\\"

    train_windows1, train_windows2, train_labels = [], [], []

    for win in range(11):  # Loop through 11 window files
        pickle_filename = f"{args.win_interval}_win{win}_addn{args.addn}_superpkt.pickle"
        pickle_path = os.path.join(prefix_pickle_output, pickle_filename)

        print(f"Loading: {pickle_path}")
        window_ingress, window_egress, circuit_labels = load_pickle_data(pickle_path)

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


# Every tor flow will have a unique threshold
def threshold_finder(input_similarity_list, curr_win, gen_ranks, thres_seed, use_global):
    output_threshold_list = []
    for simi_list_index in range (0, len (input_similarity_list)):
        correlated_similarity = input_similarity_list[simi_list_index][simi_list_index]
        temp = list (input_similarity_list[simi_list_index])
        temp.sort (reverse=True)
        cut_point = int ((len (input_similarity_list[simi_list_index]) - 1) * ((thres_seed) / 100))
        if use_global == 1:
            output_threshold_list.append (thres_seed)  # temp[cut_point]
        elif use_global != 1:
            output_threshold_list.append (temp[cut_point])  # temp[cut_point]
    return output_threshold_list


def eval_the_best_matched(tor_embs, exit_embs, similarity_threshold, single_output_l, evaluating_window,
                               last_window, correlated_shreshold, cosine_similarity_all_list, muti_output_list):
    print('single_output_l ',np.array(single_output_l).shape)
    number_of_lines = tor_embs.shape[0]
    for tor_emb_index in range(0, number_of_lines):
        t = similarity_threshold[tor_emb_index]
        constant_num = int(tor_emb_index * number_of_lines)
        for exit_emb_index in range(0, number_of_lines):
            if (cosine_similarity_all_list[tor_emb_index][exit_emb_index] >= t):
                # print('single_output_l[constant_num + exit_emb_index] ',single_output_l[constant_num + exit_emb_index])
                single_output_l[constant_num + exit_emb_index] = single_output_l[constant_num + exit_emb_index] + 1

    if (evaluating_window == last_window):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # now begin to evaluate
        # print("evaluating .......")
        for tor_eval_index in range(0, tor_embs.shape[0]):
            for exit_eval_index in range(0, tor_embs.shape[0]):
                cos_condithon_a = (tor_eval_index == exit_eval_index)
                # print((tor_eval_index * (tor_embs.shape[0])) + exit_eval_index)
                number_of_ones = (single_output_l[(tor_eval_index * (tor_embs.shape[0])) + exit_eval_index])
                # print(number_of_ones)
                cos_condition_b = (number_of_ones >= correlated_shreshold)
                cos_condition_c = (number_of_ones < correlated_shreshold)

                if (cos_condithon_a and cos_condition_b):
                    TP = TP + 1
                if (cos_condithon_a and cos_condition_c):
                    FN = FN + 1
                if ((not (cos_condithon_a)) and cos_condition_b):
                    FP = FP + 1
                if ((not (cos_condithon_a)) and cos_condition_c):
                    TN = TN + 1

        if ((TP + FN) != 0):
            TPR = (float)(TP) / (TP + FN)
        else:
            TPR = -1

        if ((FP + TN) != 0):
            FPR = (float)(FP) / (FP + TN)
        else:
            FPR = -1

        muti_output_list.append(TPR)
        muti_output_list.append(FPR)
        muti_output_list.append(calculate_bdr(TPR, FPR))
        print(TPR, FPR, calculate_bdr(TPR, FPR))
        # return single_output_l
import time

def evaluate_model(model, test_path, thr, out_put_csv,args,emb_size,use_global, muti_output_list, soft_muti_output_list):
    # Define padding sizes
    pad_t = args.tor_len * 2
    pad_e = args.exit_len * 2
    activated_windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    last_activated_window = 10
    prefix_pickle_output = ".\\Processed-test-set\\"
    single_output = []
    correlated_shreshold_value = 9
    for win in range(11):

        pickle_filename = f"{args.win_interval}_win{win}_addn{args.addn}_superpkt.pickle"
        pickle_path = os.path.join(prefix_pickle_output, pickle_filename)
        print('processing:', pickle_path)
        test_in, test_out , train_labels = load_pickle_data(pickle_path)
        # Define padding sizes
        pad_t = args.tor_len * 2
        pad_e = args.exit_len * 2

        # Apply padding
        padded_ingress = pad_windows(test_in, pad_t)
        padded_egress = pad_windows(test_out, pad_e)

        # print(f"After padding, tor shape: {padded_ingress.shape}, exit shape: {padded_egress.shape}")
        # test_in = np.array(test_in)
        # test_out = np.array(test_out)
        # print("After padding, tor shape:", test_in.shape, "exit shape:", test_out.shape)
        # Convert to PyTorch tensors and ensure correct shape
        test_in = torch.FloatTensor(padded_ingress).permute(0, 2, 1).to(device)  # (batch_size, 1, sequence_length)
        test_out = torch.FloatTensor(padded_egress).permute(0, 2, 1).to(device)
        start_time = time.time()
        in_embs = model(test_in)
        out_embs = model(test_out)


        cosine_similarity_table = cosine_similarity(in_embs.cpu().detach().numpy(), out_embs.cpu().detach().numpy())

        threshold_result = threshold_finder(cosine_similarity_table,win, 0, thr, use_global)
        if win == 0:
            for pairs in range(0, (in_embs.shape[0] * in_embs.shape[0])):
                single_output.append(0)  # ([0])
        if win == 10:
            result_matrix = np.array(single_output, dtype=object).reshape((-1, 41))
            # print("Result Matrix:\n", result_matrix)  # Print the matrix

            # # Find the highest value and corresponding index for each row
            # highest_values = np.max(result_matrix, axis=1)  # Max value per row
            # highest_indices = np.argmax(result_matrix, axis=1)  # Index of max value per row
            #
            # # Print results
            # for row_idx in range(result_matrix.shape[0]):
            #     print(f"Row {row_idx}: Highest Value = {highest_values[row_idx]}, Index = {highest_indices[row_idx]}")

        if win in activated_windows:
            eval_the_best_matched(in_embs, out_embs, threshold_result, single_output, win, last_activated_window,
                                   correlated_shreshold_value, cosine_similarity_table, muti_output_list)

    print(f"Total_inference_time. Elapsed: {time.time() - start_time:.2f}s")

    return single_output


def main():
    pad_t = args.tor_len * 2
    pad_e = args.exit_len * 2
    rank_thr_list = [60, 50, 47, 43, 40, 37, 33, 28, 24, 20, 16.667, 14, 12.5, 11, 10, 9, 8.333, 7, 6.25, 5, 4.545, 3.846, 2.941, 1.667, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    # single_output = []
    # rank_thr_list = [30]
    model_common = DFModel(input_shape=(pad_t, 1), emb_size=64, model_name='common').to(device)
    model_common.load_state_dict(torch.load(args.model1, map_location=device, weights_only=True))
    print('Sucessfully loading')
    thr = 20
    use_global = 0
    num_of_thr = len(rank_thr_list)
    rank_multi_output = []
    five_rank_multi_output = []
    epoch_index = 0
    for i in range(0, num_of_thr):
        rank_multi_output.append([(rank_thr_list[i])])
        five_rank_multi_output.append([(rank_thr_list[i])])


    for thr in rank_thr_list:
        print(epoch_index)
        single_out_put = evaluate_model(model_common,".\\Processed-test-set\\",thr,args.output,args,64,use_global,rank_multi_output[epoch_index],[])
        epoch_index = epoch_index + 1
    with open(args.output, "w", newline="") as rank_f:
        writer = csv.writer(rank_f)
        writer.writerows(rank_multi_output)
    # Convert to NumPy and reshape
    # rank_multi_output_np = np.array(rank_multi_output, dtype=object)
    print(rank_multi_output)

if __name__ == "__main__":
    main()
