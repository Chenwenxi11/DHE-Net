import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from model import ResGCN_HGT

from globel_args import device
from utils import get_metrics
from sklearn.model_selection import KFold
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch

import pickle
torch.autograd.set_detect_anomaly(True)


def trian_model2(data,y, edg_index_all, train_idx, test_idx, param):
    hidden_channels, num_heads, num_layers = (
        param.hidden_channels, param.num_heads, param.num_layers,
    )

    epoch_param = param.epochs


    model = ResGCN_HGT(hidden_channels, num_heads=num_heads, num_layers=num_layers, data=data).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)
    data_temp = copy.deepcopy(data)


    data_temp[('n1', 'e1', 'n2')].edge_index = data[('n1', 'e1', 'n2')].edge_index[:, train_idx[y[train_idx].cpu().reshape((-1,)) == 1]]

    data_temp[('n2', 'e1', 'n1')].edge_index = data[('n2', 'e1', 'n1')].edge_index[:, train_idx[y[train_idx].cpu().reshape((-1,)) == 1]]

    data_temp['x_dict'] = {ntype: data[ntype].x for ntype in data.node_types}
    edge_index_dict = {}
    for etype in data_temp.edge_types:
        edge_index_dict[etype] = data_temp[etype].edge_index

    data_temp['edge_dict'] = edge_index_dict


    auc_list = []
    model.train()
    for epoch in range(1, epoch_param+1):
        optimizer.zero_grad()
        data_temp = data_temp.to(device)
        data_temp_clone = copy.deepcopy(data_temp)
        edg_index_all = edg_index_all.to(device)
        out = model(data_temp_clone, edge_index=edg_index_all.to(device))


        # y_one_hot = F.one_hot(y[train_idx].long().squeeze(), num_classes=2).float().to(device)
        loss = F.binary_cross_entropy_with_logits(out[train_idx].to(device), y[train_idx].to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        loss = loss.item()
        if epoch % param.print_epoch == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

            model.eval()
            with torch.no_grad():

                out = model(data_temp,
                            edge_index=edg_index_all)

                out_pred_s = out[test_idx].to('cpu').detach().numpy()
                out_pred = out_pred_s
                y_true = y[test_idx].to('cpu').detach().numpy()

                auc = roc_auc_score(y_true, out_pred)
                print('AUC:', auc)


                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.other_args['arg_value'])
                auc_idx.append(epoch)
            auc_list.append(auc_idx)
            model.train()
    auc_name.extend(param.other_args['arg_name'])
    return auc_list, auc_name


def CV_train(param, args_tuple=()):
    data, y, edg_index_all = args_tuple
    idx = np.arange(y.shape[0])
    k_number = 1
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)

    kf_auc_list = []
    for train_idx,test_idx  in kf.split(idx):
        print(f'正在运行第{k_number}折, 共{k_fold}折...')
        auc_idx, auc_name = trian_model2(data, y, edg_index_all, train_idx, test_idx, param)
        k_number += 1

        kf_auc_list.append(auc_idx)


    data_idx = np.array(kf_auc_list)

    np.save(param.save_file + 'data_idx_mean.npy', data_idx)
    return kf_auc_list, auc_name