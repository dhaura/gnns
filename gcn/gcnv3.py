import torch
import torch.nn as nn
import torch.nn.functional as F
from isplib import * 
import torch_sparse

iSpLibPlugin.patch_pyg()

class GCNLayerV3(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNLayerV3, self).__init__()

        # Define input and output dimensions of the GCN layer.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the weight matrix of the layer and initialize it using uniform distribution.
        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weights)

        if use_bias:
            # Define bias and initialize it to zeros.
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            # If bias is not used, mark it as none.
            self.bias = None

    def forward(self, x, adj):
        y = torch.mm(x, self.weights) # H(l) * W(l)
        y = torch_sparse.matmul(adj, y) # A * (H(l) * W(l))
        if self.bias is not None:
            y = y + self.bias # A * (H(l) * W(l)) + B
        return y

class GCNV3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, use_bias=True):
        super(GCNV3, self).__init__()

        self.dropout = dropout
        self.gcn_layer_1 = GCNLayerV3(input_dim, hidden_dim, use_bias)
        self.gcn_layer_2 = GCNLayerV3(hidden_dim, output_dim, use_bias)

    def forward(self, x, adj):
        x = self.gcn_layer_1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn_layer_2(x, adj)
        return F.log_softmax(x, dim=1)
