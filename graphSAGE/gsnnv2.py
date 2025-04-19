import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSageLayerV2(nn.Module):
    def __init__(self, input_dim, output_dim, pool_func):
        super(GraphSageLayerV2, self).__init__()

        # Define input and output dimensions of the GCN layer.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the pooling function to be used (ex: mean, sum or max).
        self.pool_func = pool_func

        # Define a linear layer to trnasform aggregated node and neighbor fetures to output dimensions.
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        num_of_nodes = x.shape[0]
        agg_neighbors_x = torch.zeros_like(x)

        # Loop through all nodes.
        for node in range(num_of_nodes):
        neighbors = adj[node]
        if len(neighbors) > 0:
            # Aggregate features of all neighbors and the node itself.
            agg_neighbors_x[node] = self.pool_func(x[neighbors], dim=0)
        else:
            agg_neighbors_x[node] = torch.zeros_like(x[node])

        # Apply linear function to transform the aggregated features into ouput dimension.
        y = self.linear(agg_neighbors_x)
        return y

class GraphSageNNV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pool_func, droupout=0.5):
        super(GraphSageNNV2, self).__init__()

        self.dropout = droupout
        self.gsglayer1 = GraphSageLayerV2(input_dim, hidden_dim, pool_func)
        self.gsglayer2 = GraphSageLayerV2(hidden_dim, output_dim, pool_func)

    def forward(self, x, adj):
        x = self.gsglayer1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gsglayer2(x, adj)
        return F.log_softmax(x, dim=1)
