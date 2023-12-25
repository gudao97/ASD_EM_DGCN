import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ClusterGCNConv
from ED import EdgeDrop
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge


class GPModel(torch.nn.Module):
    def __init__(self, args):
        super(GPModel, self).__init__()
        self.args = args
        self.num_features = args.num_features 
        self.hidden_dim = 128
        if args.Method_GraphPool=='EdgeDrop':
            print('select:',args.Method_GraphPool)
            self.pool1 = EdgeDrop(args)
            self.pool2 = EdgeDrop(args)
            self.pool3 = EdgeDrop(args)
        else:
            print('please select pooling method')

    def forward(self, data):
        if self.args.Method_GraphPool == 'EdgeDrop':
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_attr = None
            x1 = self.pool1(x, edge_index, edge_attr, batch)
            x2 = self.pool2(x, edge_index, edge_attr, batch)
            x3 = self.pool3(x, edge_index, edge_attr, batch)
            x = F.relu(x1) + F.relu(x2) + F.relu(x3)
            print('====================================================')
            return x

class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, args):
        super(MultilayerPerceptron, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        self.selfatten = torch.nn.MultiheadAttention(embed_dim=args.num_features//3, num_heads=8, dropout=0.3)
        self.lin1 = torch.nn.Linear(self.num_features//3, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, 1)



    def forward(self,x):

        q = x[:,:384]
        k = x[:,384:768]
        v = x[:,768:]
        q = q.unsqueeze(1).permute(1,0,2)
        k = k.unsqueeze(1).permute(1,0,2)
        v = v.unsqueeze(1).permute(1,0,2)
        x,_ = self.selfatten(q,k,v)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        features = x.squeeze(0)
        x = torch.flatten(self.lin3(x))
        return x, features

class DGCN(nn.Module):
    def __init__(self, args, mode='max'):
        super(DGCN, self ).__init__()
        self.mode = mode
        self.num_features = args.num_features
        self.num_layers = 8
        self.hidden = 128
        self.dropout_ratio = args.dropout_ratio
        self.l2_regularization = args.l2_regularization
        self.conv0 = GCNConv(self.num_features, self.hidden) # GCN
        self.dropout0 = nn.Dropout(p=self.dropout_ratio)

        for i in range(1, self.num_layers):
            setattr(self, 'conv{}'.format(i), GCNConv(self.hidden, self.hidden))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=self.dropout_ratio))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = ClusterGCNConv(self.hidden, 1)
        elif mode == 'cat':
            self.fc = nn.Linear(self.num_layers * self.hidden, 1)

    def forward(self, x, edge_index, edge_weight):
        layer_out = []
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x = dropout(F.relu(conv(x, edge_index,edge_weight)))
            layer_out.append(x)
        features = F.relu(x)
        h = self.jk(layer_out)
        h = self.fc(h,edge_index)
        h = torch.flatten(h)

        l2_reg = torch.tensor(0.0).to(x.device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return h,features,self.l2_regularization * l2_reg
