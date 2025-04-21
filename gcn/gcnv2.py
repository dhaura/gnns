import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayerV2(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNLayerV2, self).__init__()

        # Define input and output dimensions of the GCN layer.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the weight matrix for neighbor features and initialize it using uniform distribution.
        self.weights_nbrs = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_nbrs)

        # Define the weight matrix for the self feature and initialize it using uniform distribution.
        self.weights_self = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_self)

        if use_bias:
            # Define bias and initialize it to zeros.
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            # If bias is not used, mark it as none.
            self.bias = None

    def forward(self, x, adj):
        y = torch.mm(x, self.weights_nbrs) # H(l) * W_nbrs(l)
        y = torch.spmm(adj, y) # A * (H(l) * W_nbrs(l))
        y = y + torch.mm(x, self.weights_self) # A * (H(l) * W_nbrs(l)) + H(l) * W_self(l)
        if self.bias is not None:
            y = y + self.bias # A * (H(l) * W_nbrs(l)) + H(l) * W_self(l) + B
        return y

class GCNV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, use_bias=True):
        super(GCNV2, self).__init__()

        self.dropout = dropout
        self.gcn_layer_1 = GCNLayerV2(input_dim, hidden_dim, use_bias)
        self.gcn_layer_2 = GCNLayerV2(hidden_dim, output_dim, use_bias)

    def forward(self, x, adj):
        x = self.gcn_layer_1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn_layer_2(x, adj)
        return F.log_softmax(x, dim=1)
