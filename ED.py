import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
import numpy as np
from ED_layers import EGNN
from torch_geometric.nn import GCNConv


class GraphRepresentation(nn.Module):

    def __init__(self, args):

        super(GraphRepresentation, self).__init__()

        self.args = args
        self.num_node_features = args.num_node_features
        self.num_edge_features = 1
        self.nhid = 128
        self.num_classes = args.num_classes
        self.edge_ratio = args.edge_ratio
        self.dropout_ratio = args.dropout_ratio
        self.enhid = args.num_edge_hidden if hasattr(args, 'num_edge_hidden') else self.nhid
    def get_scoreconvs(self):
        convs = nn.ModuleList()

        for i in range(self.args.num_convs - 1):
            conv = EGNN(self.enhid, 1)
            convs.append(conv)

        return convs

class EdgeDrop(GraphRepresentation):
    def __init__(self, args):
        super(EdgeDrop, self).__init__(args)
        self.convs = self.get_convs()
        self.EGNN = self.get_convs(conv_type='Hyper')[:-1]
        self.scoreconvs = self.get_scoreconvs()

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), device=edge_index.device)
        xs = 0
        for _ in range(self.args.num_convs):
            edge_batch = edge_index[0].reshape(-1)
            if _ == 0:
                x = F.relu(self.convs[_](x, edge_index))
            else:
                x = F.relu(self.convs[_](x, edge_index, edge_weight))
            if _ < self.args.num_convs - 1:
                edge_attr = F.relu(self.EGNN[_](edge_attr, edge_index))
                score = torch.tanh(self.scoreconvs[_](edge_attr, edge_index).squeeze())
                if _ == 1:
                    sorted_scores, indices = torch.sort(score, descending=True)
                    top_10_max_values = sorted_scores[:10]
                    top_10_indices = indices[:10]
                    if top_10_max_values[0] != 0:
                        print(top_10_max_values[1:])
                    top10 = top_10_indices.cpu().detach().numpy()
                    with open('top10.csv','a') as file:
                        np.savetxt(file, top10, delimiter=',',fmt='%.0f')
                    print(top_10_max_values)
                    print(top_10_indices)
                perm = topk(score, self.edge_ratio, edge_batch)
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm, :]
                edge_weight = score[perm]
                edge_weight = torch.clamp(edge_weight, min=0, max=1)
            xs += torch.cat([gmp(x, batch), gap(x, batch), gsp(x, batch)], dim=1)
        x = xs.squeeze(1)
        return x

    def get_convs(self, conv_type='GCN'):
        convs = nn.ModuleList()
        for i in range(self.args.num_convs):
            if conv_type == 'GCN':
                if i == 0:
                    conv = GCNConv(self.num_node_features, self.nhid)
                else:
                    conv = GCNConv(self.nhid, self.nhid)
            elif conv_type == 'Hyper':
                if i == 0:
                    conv = EGNN(self.num_edge_features, self.enhid)
                else:
                    conv = EGNN(self.enhid, self.enhid)
            else:
                raise ValueError("Conv Name <{}> is Unknown".format(conv_type))
            convs.append(conv)
        return convs

