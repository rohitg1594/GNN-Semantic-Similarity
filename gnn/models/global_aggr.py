import torch
from torch_geometric.nn import GlobalAttention
from torch_scatter import scatter_mean, scatter_max, scatter_add
from gnn.models.mlp import MLP


class GlobalAggregator(torch.nn.Module):
    def __init__(self, args):
        super(GlobalAggregator, self).__init__()
        self.aggr = args.aggr
        if self.aggr == "global-attn":
            mlp1 = MLP(args.num_layers, args.input_size, args.hidden_size, 1)
            mlp2 = MLP(args.num_layers, args.input_size, args.hidden_size, args.output_size)
            self.global_attention = GlobalAttention(mlp1, mlp2)

    def forward(self, x, data):
        if self.aggr == 'mean':
            x = scatter_mean(x, data.batch, dim=0)
        if self.aggr == 'max':
            x, _ = scatter_max(x, data.batch, dim=0)
        if self.aggr == 'sum':
            x = scatter_add(x, data.batch, dim=0)
        if self.aggr == 'mean-main':
            x = x * data.main_chain_mask.unsqueeze(1).expand_as(x)
            x = scatter_mean(x, data.batch, dim=0)
        if self.aggr == 'global-attn':
            x = self.global_attention(x, data.batch)

        return x
