import time
import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import KFold
from algorithm.Properties import Properties
from algorithm.Novel import Novel
from algorithm.LsncAnn import Trainer
from algorithm.LsncAnn import Tester

def read_data(train_filename: str = ""):
    # Read mat data.
    all_data_read = sio.loadmat(train_filename)
    train_data_read = np.array(all_data_read['train_data'])
    test_data_read = np.array(all_data_read['test_data'])
    train_targets_read = np.array(all_data_read['train_target']).transpose()
    test_targets_read = np.array(all_data_read['test_target']).transpose()
    train_targets_read[np.where(train_targets_read < 0.0)] = 0.0
    test_targets_read[np.where(test_targets_read < 0.0)] = 0.0
    train_data_read = (train_data_read - train_data_read.min(axis=0)) / \
                           (train_data_read.max(axis=0) - train_data_read.min(axis=0) + 0.0001)
    test_data_read = (test_data_read - test_data_read.min(axis=0)) / \
                          (test_data_read.max(axis=0) - test_data_read.min(axis=0) + 0.0001)

    temp_sum = np.sum(train_targets_read)
    area_train = train_targets_read.size
    ones_train = (temp_sum + area_train) / 2
    proportion = ones_train / area_train

    print("Proportion of 1 in train target (label matrix): ", ones_train, " out of ", area_train,
          " gets ", proportion)

    return train_data_read, train_targets_read, test_data_read, test_targets_read
def for_kf(kf_num: int = 5, index: int = 10):
    list = ['Emotion']
    for data in list:
        start = time.time()
        dataset = Properties(data)
        train_data, train_label, test_data, test_label = read_data(dataset.filename)
        data = np.vstack((train_data, test_data))
        label = np.vstack((train_label, test_label))

        tr = Trainer()
        te = Tester()
        metrics_dict = {"Peak F1-Score": [], "NDCG": [], "Macro-AUC": [], "Micro-AUC": [],
                        "Coverage": [], "One Error": [], "Ranking Loss": [], "Hamming Loss": []}

        kf = KFold(kf_num, shuffle=True)
        for k, (train_index, test_index) in enumerate(kf.split(data)):
            train_data = data[train_index, :]
            test_data = data[test_index, :]
            train_label = label[train_index, :]
            test_label = label[test_index, :]
            norkel_ann = Novel(train_data, train_label,
                   test_data, test_label,
                   dataset.k_label, dataset.parallel_layer_num_nodes).to(torch.device('cuda'))
            # training #
            tr.initialization(dataset, train_data, train_label,
                              norkel_ann.train_target_loss,
                              norkel_ann.train_target_list_loss,
                              dataset.parallel_layer_num_nodes,
                              dataset.k_label,norkel_ann.label_select)

            parallel_out_put = tr.run_train()
            # testing #
            te.initialization(dataset, test_data, test_label, norkel_ann.label_select)
            te.run_test()

        end = time.time()
        Runtime = end - start
        print('程序运行时间:{0} is {1}'.format(data, Runtime))

if __name__ == "__main__":
    for_kf()
