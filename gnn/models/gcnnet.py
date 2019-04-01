import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean, scatter_max, scatter_add


class GCNNet(torch.nn.Module):
    def __init__(self,
                 aggr='mean',
                 num_embs=None,
                 hidden_size=256,
                 input_size=128,
                 output_size=128,
                 num_layers=2,
                 dropout=0.5):
        super(GCNNet, self).__init__()
        if num_embs is None:
            print("Must pass in the number of embeddings")
            exit()
        self.embed = nn.Embedding(num_embs, input_size)

        self.in_layer = GCNConv(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([GCNConv(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.out_layer = GCNConv(hidden_size, output_size)

        self.num_layers = num_layers
        self.aggr = aggr
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def forward(self, data):
        if isinstance(data, list):
            data = data[0]

        edge_index, x_ids = data.edge_index, data.node_ids
        # print(f"edge index: {edge_index.get_device()}")
        # print(f"x ids: {x_ids.get_device()}")
        # print(f"model embed: {self.embed.weight.get_device()}")

        x = self.embed(x_ids)
        x = self.in_layer(x, edge_index)
        x = self.relu(x)
        x = self.dp(x)
        for layer in self.hidden_layers:
            x = layer(x, edge_index) + x
            x = self.relu(x)
            x = self.dp(x)
        x = self.out_layer(x, edge_index)

        if self.aggr == 'mean':
            x = scatter_mean(x, data.batch, dim=0)
        if self.aggr == 'max':
            x, _ = scatter_max(x, data.batch, dim=0)
        if self.aggr == 'sum':
            x = scatter_add(x, data.batch, dim=0)

        return x

    def __str__(self):
        return f"In Layer: {self.in_layer}, " + \
              f"Hiddin Layers: {self.hidden_layers}, " + \
              f"Out Layer: {self.out_layer}"


if __name__ == "__main__":

    for l in range(2, 5):
        net = GCNNet(num_embs=10, num_layers=l)
