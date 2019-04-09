import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv
from gnn.models.global_aggr import GlobalAggregator


class GatedConv(torch.nn.Module):
    def __init__(self,
                 aggr='mean',
                 num_embs=None,
                 hidden_size=256,
                 input_size=128,
                 output_size=128,
                 num_layers=2,
                 dropout=0.5,
                 args=None):
        super(GatedConv, self).__init__()
        if num_embs is None:
            print("Must pass in the number of embeddings")
            exit()
        self.embed = nn.Embedding(num_embs, input_size)

        self.gated_conv = GatedGraphConv(output_size, num_layers)

        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)
        self.global_aggr = GlobalAggregator(args)

    def forward(self, data):
        if isinstance(data, list):
            data = data[0]

        edge_index, x_ids = data.edge_index, data.node_ids

        x = self.embed(x_ids)
        x = self.gated_conv(x, edge_index)
        x = self.global_aggr(x, data)

        return x
