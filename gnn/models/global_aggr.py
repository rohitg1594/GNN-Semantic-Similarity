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

        self.registry = {'mean': self.mean,
                         'max': self.max,
                         'sum': self.sum,
                         'mean-main': self.mean_main,
                         'mean-side': self.mean_side,
                         'global-attn': self.global_attn}

    def forward(self, x, data):
        if self.aggr in self.registry:
            return self.registry[self.aggr](x, data)
        else:
            print(f"Aggregation {self.aggr} not recognized.")
            raise NotImplementedError

    @staticmethod
    def mean(x, data):
        return scatter_mean(x, data.batch, dim=0)

    @staticmethod
    def max(x, data):
        return scatter_max(x, data.batch, dim=0)

    @staticmethod
    def sum(x, data):
        return scatter_add(x, data.batch, dim=0)

    @staticmethod
    def mean_main(x, data):
        x = x * data.main_chain_mask.unsqueeze(1).expand_as(x)
        return scatter_mean(x, data.batch, dim=0)

    @staticmethod
    def mean_side(x, data):
        main_chain_mask = data.main_chain_mask.unsqueeze(1).expand_as(x)
        side_chain_mask = main_chain_mask.eq(0).float()
        x = x * side_chain_mask
        return scatter_mean(x, data.batch, dim=0)

    def global_attn(self, x, data):
        return self.global_attention(x, data.batch)


if __name__ == "__main__":
    class dummy:
        def __init__(self):
            pass

    args = dummy()
    args.agg = 'mean-side'

    x = torch.randn(3, 4, 5)
    data = dummy()
