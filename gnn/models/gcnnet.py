from gnn.models.global_aggr import GlobalAggregator
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self,
                 aggr='mean',
                 num_embs=None,
                 hidden_size=256,
                 input_size=128,
                 output_size=128,
                 num_layers=2,
                 dropout=0.5,
                 args=None):
        super(GCNNet, self).__init__()
        if num_embs is None:
            print("Must pass in the number of embeddings")
            exit()
        self.embed = nn.Embedding(num_embs, input_size)

        self.in_layer = GCNConv(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([GCNConv(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.out_layer = GCNConv(hidden_size, output_size)

        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)
        self.global_aggr = GlobalAggregator(args)

    def forward(self, data):
        if isinstance(data, list):
            data = data[0]

        edge_index, x_ids = data.edge_index, data.node_ids

        x = self.embed(x_ids)
        x = self.in_layer(x, edge_index)
        x = self.relu(x)
        x = self.dp(x)
        for layer in self.hidden_layers:
            x = layer(x, edge_index) + x
            x = self.relu(x)
            x = self.dp(x)
        x = self.out_layer(x, edge_index)

        x = self.global_aggr(x, data)

        return x
