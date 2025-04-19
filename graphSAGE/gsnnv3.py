import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class GraphSageLayerV3(nn.Module):
  def __init__(self, input_dim, output_dim, pool_method):
    super(GraphSageLayerV3, self).__init__()

    # Define input and output dimensions of the GCN layer.
    self.input_dim = input_dim
    self.output_dim = output_dim

    # Define the pooling function to be used (ex: mean, sum or max).
    self.pool_method = pool_method

    # Define a linear layer to trnasform concatenated node and aggregated neighbor fetures to output dimensions.
    self.linear = nn.Linear(input_dim * 2, output_dim)

  def forward(self, x, edge_index):
    # Extract source and destination indexes.
    src, dst = edge_index
    # Extrect neighbour features of sources of every edge.
    neighbors_x = x[src]

    # Aggregate features of all neighbors.
    agg_neighbors_x = scatter(neighbors_x, dst, dim=0, dim_size=x.size(0), reduce=self.pool_method)

    # Concatenate aggregated neighbor features with node features.
    y = torch.cat([x, agg_neighbors_x], dim=1)

    # Apply linear function to transform the concatenated features into ouput dimension.
    y = self.linear(y)
    return y


class GraphSageNNV3(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, pool_method, droupout=0.5):
    super(GraphSageNNV3, self).__init__()

    self.dropout = droupout
    self.gsglayer1 = GraphSageLayerV3(input_dim, hidden_dim, pool_method)
    self.gsglayer2 = GraphSageLayerV3(hidden_dim, output_dim, pool_method)

  def forward(self, x, adj):
    x = self.gsglayer1(x, adj)
    x = F.relu(x)
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gsglayer2(x, adj)
    return F.log_softmax(x, dim=1)
