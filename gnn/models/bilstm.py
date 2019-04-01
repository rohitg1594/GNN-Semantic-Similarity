# BiLSTM baseline model
import torch.nn as nn
import torch

from logging import getLogger

logger = getLogger(__name__)


class BILSTM(nn.Module):
    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 output_size=None,
                 num_embs=None,
                 num_layers=2,
                 dropout=0.3,
                 aggr=None,):
        super(BILSTM, self).__init__()
        if num_embs is None:
            logger.error("Must provide number of embeddings to the model, exiting......")
            exit()
        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              batch_first=True,
                              bidirectional=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(num_embs, input_size)

    def forward(self, input):
        x, lens = input
        b = x.shape[0]  # batch size
        output, _ = self.bilstm(self.emb(x),)

        return output[torch.arange(b), lens, :]
