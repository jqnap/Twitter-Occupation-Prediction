import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, flag_output=False):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)

        self.dropout = dropout
        self.flag_output = flag_output

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)

        return F.log_softmax(x, dim=1)
