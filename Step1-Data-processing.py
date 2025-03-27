import os

def find_key(input_dict, value):
    return {k for k, v in input_dict.items() if v == value}

def parse_csv(csv_path, interval, final_names, threshold):
    ingress_path = csv_path + 'inflow'
    egress_path = csv_path + 'outflow'
    print(ingress_path, egress_path, interval)

    file_names = []
    for txt_file in os.listdir(ingress_path):
        file_names.append(txt_file)

    # print(file_names)
    for i in range(len(file_names)):
        ## ingress windows generation
        with open(ingress_path+'\\'+file_names[i]) as f:
            in_lines = []
            full_lines =f.readlines()
            for line in full_lines: # break the session into bursts
                time = float(line.split('\t')[0])
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                in_lines.append(line)

        with open(egress_path + '\\'+file_names[i]) as f:
            out_lines = []
            full_lines = f.readlines()
            for line in full_lines:
                time = float(line.split('\t')[0])
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                out_lines.append(line)

        if (len (in_lines) > threshold) and (len(out_lines) > threshold):
            # print('in_line_nuber of packets : ', len(in_lines))
            # print('out_line_nuber of packets : ', len(out_lines))
            if file_names[i] in final_names.keys():
                final_names[file_names[i]] += 1
            else:
                final_names[file_names[i]] = 1

    # for x in final_names:
    #     print(x,final_names[x]) # file_name with number of windows in this file, each window with number of packets larger than threshold


def create_overlap_windows_csv(csv_path, out_path, threshold, interval, num_windows, add_num):
    global final_names
    final_names = {}

    file_write = open(out_path,'w+')
    for win in range(num_windows):
        parse_csv(csv_path, [win*add_num, win*add_num+interval],final_names,threshold)

    for name in list(find_key(final_names,num_windows)):
        file_write.write(name)
        file_write.write('\n')
    file_write.close()

# data_path = '.\\CrawlE_Proc\\'
data_path =  '.\\Evaluation_data\\'
out_file_path = 'qualified_file_test.txt'
# out_file_path = 'qualified_file_training.txt'
threshold = 10
interval = 5
num_windows = 10
add_num = 2

create_overlap_windows_csv(data_path,out_file_path,threshold,interval,num_windows,add_num)