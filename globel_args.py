import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os


def set_attr(config, param_search):
    param_grid = param_search
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for keys in param_keys:
            setattr(config, keys, param[keys])
            config.other_args['arg_name'].append(keys)
            config.other_args['arg_value'].append(param[keys])
        yield config
    return 0


def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

