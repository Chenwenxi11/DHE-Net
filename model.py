import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HGTConv, Linear
from torch.nn import Module
from sklearn.decomposition import PCA

class ResGCN(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.conv3 = GCNConv(out_channels, out_channels)
        self.residual = Linear(in_channels, out_channels)
        self.pca = PCA(n_components=256)

    def forward(self, x, edge_index):
        device = edge_index.device


        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[DEBUG] NaN or Inf detected in input x: {x}")

        if x.dim() == 2 and x.shape[0] == x.shape[1]:
            x = x.cpu()
            x = self.pca.fit_transform(x)
            x = torch.tensor(x).to(device).float()

        assert x.is_cuda, "x is not on CUDA"
        res = self.residual(x.clone().to(device))

        assert res.is_cuda, "res is not on CUDA"


        x = self.conv1(x.contiguous().clone().to(device), edge_index).relu()

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[DEBUG] NaN or Inf detected after conv1: {x}")


        x = self.conv2(x.contiguous().clone().to(device), edge_index).relu()

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[DEBUG] NaN or Inf detected after conv2: {x}")


        x = self.conv3(x.contiguous().clone().to(device), edge_index).relu()

        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[DEBUG] NaN or Inf detected after conv3: {x}")

        return x + res

class HGT(Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(-1, hidden_channels, data.metadata(), num_heads)
            self.convs.append(conv)

        self.fc = Linear(hidden_channels * 2, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data, edge_index):
        x_dict_, edge_index_dict = data['x_dict'], data['edge_dict']
        x_dict = x_dict_.copy()


        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu()

            # 检查 NaN 或 Inf
            if torch.isnan(x_dict[node_type]).any() or torch.isinf(x_dict[node_type]).any():
                print(f"[DEBUG] NaN or Inf detected in node_type {node_type}: {x_dict[node_type]}")

        all_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_list.append(x_dict.copy())


        for i, _ in x_dict_.items():
            x_dict[i] = torch.cat(tuple(x[i] for x in all_list), dim=1)

        m_index = edge_index[0]
        d_index = edge_index[1]

        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])
        y = Em @ Ed.t()
        y = y[m_index, d_index].unsqueeze(-1)


        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[DEBUG] NaN or Inf detected in final output y: {y}")

        return y

class ResGCN_HGT(Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):
        super().__init__()
        self.resgcn = ResGCN(hidden_channels, hidden_channels)
        self.hgt = HGT(hidden_channels, num_heads, num_layers, data)

    def forward(self, data, edge_index):

        resgcn_out = self.resgcn(data['x_dict']['n1'], edge_index)


        data['x_dict']['n1'] = resgcn_out

        out = self.hgt(data, edge_index=edge_index)


        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[DEBUG] NaN or Inf detected in final output of ResGCN_HGT: {out}")

        return out
