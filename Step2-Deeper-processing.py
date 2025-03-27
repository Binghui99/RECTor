import os
import numpy as np
import pickle


def parse_csv(csv_path, interval, file_names):
    ingress_path = csv_path + 'inflow'
    egress_path = csv_path + 'outflow'

    print(ingress_path,egress_path,interval)

    ingress = []
    egress =[]

    ingress_len = []
    egress_len = []

    flow_count = 0
    final_names =[]

    num_in_big_packets_count = []
    num_out_big_packets_count = []


    for i in range(len(file_names)):
        in_flows = []
        out_flows = []

        with open(ingress_path + '\\' + file_names[i]) as f:
            pre_in_time = 0.0
            full_in_lines=f.readlines()

            if len(full_in_lines) == 0:
                continue
            big_pkt =[]
            num_in_big_packets = 0
            for line in full_in_lines:
                time_size_together = []
                arrive_time = float(line.split('\t')[0])
                size = float(line.split('\t')[1])

                if size>0:
                    iat = arrive_time - pre_in_time
                else:
                    iat = -(arrive_time - pre_in_time)

                if float(arrive_time > interval[1]):
                    break
                if float(arrive_time < interval[0]):
                    continue

                if abs(size) > 512: #  Only process packets larger than 512 bytes (removes ACK packets).
                    if (pre_in_time != 0) and (iat == 0):
                        big_pkt.append(size)
                        continue
                    if len(big_pkt) != 0 :
                        last_pkt = in_flows.pop()
                        in_flows.append({'iat': last_pkt['iat'], 'size': sum(big_pkt)+big_pkt[0]})
                        big_pkt = []
                        num_in_big_packets += 1
                    time_size_together.append(iat)
                    time_size_together.append(size)
                    time_size_together = np.array(time_size_together)
                    in_flows.append({'iat': time_size_together[0], 'size': time_size_together[1]})
                    pre_in_time = arrive_time

        with open(egress_path + '\\' + file_names[i]) as f:
            pre_out_time = 0.0
            full_out_lines=f.readlines()
            if len(full_out_lines) == 0:
                continue
            big_pkt = []
            num_out_big_packets = 0

            for line in full_out_lines:
                time_size_together = []
                arrive_time = float(line.split('\t')[0])
                size = float(line.split('\t')[1])

                if size > 0:
                    iat = arrive_time - pre_out_time
                else:
                    iat = -(arrive_time - pre_out_time)

                if float(arrive_time > interval[1]):
                    break
                if float(arrive_time < interval[0]):
                    continue

                if abs(size) > 66:
                    if (pre_out_time != 0) and (iat == 0):
                        big_pkt.append(size)
                        continue
                    if len(big_pkt) != 0:
                        last_pkt = out_flows.pop()
                        out_flows.append({'iat' : last_pkt['iat'], 'size': sum(big_pkt)+big_pkt[0]})
                        big_pkt = []
                        num_out_big_packets += 1
                    time_size_together.append(iat)
                    time_size_together.append(size)
                    time_size_together = np.array(time_size_together)
                    out_flows.append({'iat': time_size_together[0], 'size': time_size_together[1]} )
                    pre_out_time = arrive_time

        if (len(in_flows) != 0) and (len(out_flows) != 0):
            ingress_len.append(len(in_flows))
            num_in_big_packets_count.append(num_in_big_packets)
            egress_len.append(len(out_flows))
            num_out_big_packets_count.append(num_out_big_packets)

            ingress.append(in_flows)
            egress.append(out_flows)
            final_names.append(file_names[i])
            flow_count += 1

    print(interval, 'mean', np.mean(np.array(ingress_len)), np.mean(np.array(egress_len)), np.mean(num_in_big_packets_count),
          np.mean(num_out_big_packets_count), flow_count)
    print(len(ingress), len(egress), len(final_names))
    # print(file_names)
    return ingress, egress, final_names

def create_overlap_window_csv(csv_path, file_list, prefix_file_output, interval, num_windows, add_num):
    window_seq = []
    with open(file_list, 'r') as f:
        file_names = [line.strip() for line in f if line.strip()]

    # ensure output path
    output_dir = os.path.dirname(prefix_file_output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for win in range(num_windows):
        current_interval = [win * add_num, win*add_num+interval]

        ingress, egress, labels = parse_csv(csv_path, current_interval, file_names)

        window_data = {"ingress": ingress, "egress": egress, "label":labels}
        window_seq.append(window_data)


        # create the output files use f-string formating

        output_file = f"{prefix_file_output}{interval}_win{win}_addn{add_num}_superpkt.pickle"

        with open(output_file,'wb') as handle:
            pickle.dump(window_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return np.array(window_seq)


# data_path = ".\\DeepCoffea-data\\"

# file_list_path = "qualified_file_training.txt"
# prefix_pickle_output =".\\Processed-window-files\\"
# #
# data_path = '.\\CrawlE_Proc\\'
# out_file_path = 'qualified_file_training.txt'

#test

data_path =  '.\\Evaluation_data\\'
file_list_path  = 'qualified_file_test.txt'
prefix_pickle_output =".\\Processed-test-set\\"

create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output,5,10,2)








