import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from globel_args import device
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import os
from utils import get_data
from train_model import CV_train
from torch_geometric.nn import GCNConv, HGTConv, Linear
from torch.nn import Module

from sklearn.decomposition import PCA

torch.autograd.set_detect_anomaly(True)


def set_seed(seed):
    torch.manual_seed(seed)

    # np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    def __init__(self):
        self.datapath = './data/'
        self.save_file = './save_file/'

        self.kfold = 5
        self.maskMDI = False
        self.hidden_channels = 512  # 256 512
        self.num_heads = 8  # 4 8
        self.num_layers = 4  # 4 8
        self.self_encode_len = 256
        self.globel_random = 120
        self.other_args = {'arg_name': [], 'arg_value': []}


        self.epochs = 500  ## 1000
        self.print_epoch = 20  ## 20


def set_attr(config, param_search):
    param_grid = param_search
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for keys in param_keys:
            setattr(config, keys, param[keys])
            config.other_args['arg_name'].append(keys)
            print(keys, param[keys])
            config.other_args['arg_value'].append(param[keys])
        yield config
    return 0


class Data_paths:
    def __init__(self):
        self.paths = './data/'
        self.md = self.paths + 'circ-dis_ass.csv'
        self.mm = [self.paths + 'c_gs.csv', self.paths + 'c_ss.csv']
        self.dd = [self.paths + 'd_gs.csv', self.paths + 'd_ss.csv']


best_param_search = {
    'hidden_channels': [64, 128, 256, 512],
    'num_heads': [4, 8, 16, 32],
    'num_layers': [2, 4, 6, 8],
    # 'CL_margin' :[0.5,1.0,1.5,2.0],
    'CL_noise_max': [0.05, 0.1, 0.2, 0.4],
}

best_param_search = {
    'hidden_channels': [256],
    'num_heads': [8],
    'num_layers': [6],
    # 'CL_margin' :[0.5,1.0,1.5,2.0],
    'CL_noise_max': [0.1],
}

if __name__ == '__main__':

    set_seed(521)
    param_search = best_param_search

    save_file = '5cv_data_1000'
    params_all = Config()
    param_generator = set_attr(params_all, param_search)
    data_list = []
    filepath = Data_paths()

    while True:
        try:
            params = next(param_generator)
        except StopIteration:
            break

        print(f"\n[DEBUG] Using parameters: {params.__dict__}")


        data_tuple = get_data(file_pair=filepath, params=params)
        print(f"[DEBUG] Data loaded, checking for NaN/Inf in data...")

        data, y, edg_index_all = data_tuple





        if isinstance(data, HeteroData):

            for key, value in data.items():
                if isinstance(value, dict):

                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            if torch.any(torch.isnan(sub_value)) or torch.any(torch.isinf(sub_value)):
                                print(f"[ERROR] Found NaN or Inf in {key}->{sub_key}!")
                        elif isinstance(sub_value, np.ndarray):
                            sub_value_tensor = torch.tensor(sub_value)
                            if torch.any(torch.isnan(sub_value_tensor)) or torch.any(torch.isinf(sub_value_tensor)):
                                print(f"[ERROR] Found NaN or Inf in {key}->{sub_key} (converted from np.ndarray)!")
                elif isinstance(value, torch.Tensor):
                    if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                        print(f"[ERROR] Found NaN or Inf in {key}!")
                elif isinstance(value, np.ndarray):
                    value_tensor = torch.tensor(value)
                    if torch.any(torch.isnan(value_tensor)) or torch.any(torch.isinf(value_tensor)):
                        print(f"[ERROR] Found NaN or Inf in {key} (converted from np.ndarray)!")
        else:

            if isinstance(data, torch.Tensor):
                if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)):
                    print("[ERROR] Found NaN or Inf in data!")
            elif isinstance(data, np.ndarray):
                data_tensor = torch.tensor(data)
                if torch.any(torch.isnan(data_tensor)) or torch.any(torch.isinf(data_tensor)):
                    print("[ERROR] Found NaN or Inf in data (converted from np.ndarray)!")


        if isinstance(y, torch.Tensor):
            if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                print("[ERROR] Found NaN or Inf in labels!")
        elif isinstance(y, np.ndarray):
            y_tensor = torch.tensor(y)
            if torch.any(torch.isnan(y_tensor)) or torch.any(torch.isinf(y_tensor)):
                print("[ERROR] Found NaN or Inf in labels (converted from np.ndarray)!")


        if isinstance(edg_index_all, torch.Tensor):
            if torch.any(torch.isnan(edg_index_all)) or torch.any(torch.isinf(edg_index_all)):
                print("[ERROR] Found NaN or Inf in edge index!")
        elif isinstance(edg_index_all, np.ndarray):
            edg_index_all_tensor = torch.tensor(edg_index_all)
            if torch.any(torch.isnan(edg_index_all_tensor)) or torch.any(torch.isinf(edg_index_all_tensor)):
                print("[ERROR] Found NaN or Inf in edge index (converted from np.ndarray)!")


        data_tuple = (data.to(device), y.to(device), edg_index_all.to(device))


        try:
            print("[DEBUG] Starting cross-validation training...")
            data_idx, auc_name = CV_train(params, data_tuple)  # 交叉验证
            if data_idx is not None:
                data_list.append(data_idx)
            else:
                print(f"Warning: No data returned for params: {params}")

        except Exception as e:
            print(f"[ERROR] Error in CV_train: {str(e)}")

    if len(data_list) > 1:
        data_all = np.concatenate(tuple(x for x in data_list), axis=1)
    else:
        data_all = data_list[0]


    try:
        np.save(params_all.save_file + save_file + '.npy', data_all)
        print("[DEBUG] Data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Error saving data: {str(e)}")

    print(f"[DEBUG] AUC name: {auc_name}")

    try:
        data_idx = np.load(params_all.save_file + save_file + '.npy', allow_pickle=True)
        print("[DEBUG] Data loaded for final processing.")
    except Exception as e:
        print(f"[ERROR] Error loading saved data: {str(e)}")


    data_mean = data_idx[:, :, 2:].mean(0)
    idx_max = data_mean[:, 1].argmax()
    print("\n[DEBUG] 最大值为：")
    print(data_mean[idx_max, :])
