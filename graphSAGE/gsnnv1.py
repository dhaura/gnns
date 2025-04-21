import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSageLayerV1(nn.Module):
    def __init__(self, input_dim, output_dim, pool_func):
        super(GraphSageLayerV1, self).__init__()

        # Define input and output dimensions of the GCN layer.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the pooling function to be used (ex: mean, sum or max).
        self.pool_func = pool_func

        # Define a linear layer to trnasform concatenated node and aggregated neighbor fetures to output dimensions.
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, adj):
        num_of_nodes = x.shape[0]
        agg_neighbors_x = torch.zeros_like(x)

        # Loop through all nodes.
        for node in range(num_of_nodes):
            neighbors = adj[node]
            if len(neighbors) > 0:
                # Aggregate features of all neighbors.
                agg_neighbors_x[node] = self.pool_func(x[neighbors], dim=0)
            else:
                agg_neighbors_x[node] = torch.zeros_like(x[node])

        # Concatenate aggregated neighbor features with node features.
        y = torch.cat([x, agg_neighbors_x], dim=1)
        # Apply linear function to transform the concatenated features into ouput dimension.
        y = self.linear(y)
        return y

class GraphSageNNV1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pool_func, droupout=0.5):
        super(GraphSageNNV1, self).__init__()

        self.dropout = droupout
        self.gsglayer1 = GraphSageLayerV1(input_dim, hidden_dim, pool_func)
        self.gsglayer2 = GraphSageLayerV1(hidden_dim, output_dim, pool_func)

    def forward(self, x, adj):
        x = self.gsglayer1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gsglayer2(x, adj)
        return F.log_softmax(x, dim=1)
