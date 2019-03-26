import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_scatter import scatter_mean

from fairseq.gnn.models.mlp import MLP


class GINNet(torch.nn.Module):
    def __init__(self, aggr='mean', num_embs=None, hidden_dim=256, input_dim=128, output_dim=128, num_layers=2, num_mlp_layers=2, dropout=0.5, train_eps=False):
        super(GINNet, self).__init__()
        if num_embs is None:
            print("Must pass in the number of embeddings")
            exit()
        self.embed = nn.Embedding(num_embs, input_dim)

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batch-norms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        # List of GINs
        self.gins = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.gins.append(GINConv(self.mlps[i], train_eps=train_eps))

        self.num_layers = num_layers
        self.aggr = aggr
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)

    def forward(self, data):
        edge_index, x_ids = data.edge_index, data.node_ids
        x = self.embed(x_ids)
        # print(f"X[0]: {x.shape}")

        hidden_reps = [x]
        for i in range(self.num_layers - 1):
            x = self.gins[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = self.relu(x)
            # print(f"X[{i + 1}]: {x.shape}")
            hidden_reps.append(x)

        score_over_layer = 0
        for layer, h in enumerate(hidden_reps):
            score_over_layer += self.dp(self.linears_prediction[layer](h))

        if self.aggr == 'mean':
            score_over_layer = scatter_mean(score_over_layer, data.batch, dim=0)
        # TODO: Implement sum and max pool

        return score_over_layer


if __name__ == "__main__":

    for l in range(2, 5):
        net = GINNet(num_embs=10, num_layers=l)
