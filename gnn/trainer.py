import torch
import numpy as np
import torch.nn.functional as F
from gnn.utils import send_to_device
from torch_geometric.data import Data


class Trainer:

    def __init__(self, **kwargs):

        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.num_layers = kwargs['num_layers']
        self.dp = kwargs['dp']
        self.param2perf = kwargs['param2perf']
        self.logger = kwargs['logger']
        self.model = kwargs['model']
        self.args = kwargs['args']
        self.loader = kwargs['loader']
        self.validator = kwargs['validator']
        self.device = kwargs['device']
        self.dictionary = kwargs['dictionary']
        optim = kwargs['optimizer']
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            self.logger.error(f"optim: {optim} not recognized, exiting.....")
            raise AttributeError

        if self.param2perf is not None:
            key = f"LR: {self.lr}, WD: {self.wd}, NUM_LAYERS: {self.num_layers}, DP: {self.dp}"
            self.logger.info(key)
            self.key_params = dict([param.strip().split(':') for param in key.split(',')])

    def train(self):

        self.logger.info(f"Model: {self.model}")
        self.logger.info("Starting Training....")

        for epoch in range(self.args.num_epochs):

            self.model.train()
            train_losses = []
            for i, (inp_src, inp_tgt, y) in enumerate(self.loader):

                self.optimizer.zero_grad()
                inp_src = send_to_device(inp_src, self.device)
                inp_tgt = send_to_device(inp_tgt, self.device)
                out_src = self.model(inp_src)
                out_tgt = self.model(inp_tgt)
                scores = torch.sigmoid((out_src * out_tgt).sum(dim=1))  # dot product

                if self.args.verbose == 2:
                    self.log_verbose2(inp_src, inp_tgt, out_src, out_tgt, y)

                if i % 100 == 0 and self.args.verbose >= 1:
                    self.log_verbose1(scores, y)

                if not (((scores >= 0.) & (scores <= 1.)).all() and ((y >= 0.) & (y <= 1.)).all()):
                    raise OverflowError
                else:
                    loss = self.optimizer_step(scores, y)
                    train_losses.append(loss)

            results = self.validator.validate(self.model, verbose=self.args.verbose)
            valid_loss = results['Valid Loss']
            train_loss = np.mean(train_losses)

            result_str = self.create_result_str(results, train_loss, valid_loss, epoch)
            self.logger.info(result_str)

            if self.param2perf is not None:
                self.update_param2perf(results, train_loss, epoch)

        self.logger.info("Training complete.")

        return self.model, self.optimizer, self.param2perf

    def update_param2perf(self, results, train_loss, epoch):
        res_d = {**results,
                 'Train Loss': train_loss,
                 'Epoch': epoch}
        save_d = {**self.key_params, **res_d}
        self.param2perf.append(save_d)

    def log_verbose2(self, inp_src, inp_tgt, out_src, out_tgt, y):
        if isinstance(inp_src[0], Data):
            inp_src = inp_src[0].node_ids
            inp_tgt = inp_tgt[0].node_ids
        else:
            inp_src = inp_src[0]
            inp_tgt = inp_tgt[0]
        print(f"SRC: {inp_src}")
        print(f"TGT: {inp_tgt}")
        self.logger.info(f"SOURCE : \n {self.dictionary.string(inp_src)}, shape: {inp_src.shape}")
        self.logger.info(f"TARGET : \n {self.dictionary.string(inp_tgt)}, shape: {inp_tgt.shape}")
        self.logger.info(f"Y: {y}")
        self.logger.info(f"Out en: {out_src.shape}, Out de : {out_tgt.shape}")
        exit()

    def log_verbose1(self, scores, y):
        self.logger.info(f"TRAIN - i: {i}")
        self.logger.info(f" SCORES: Mean: {torch.mean(scores)}, STD: {torch.std(scores)} Shape: {scores.shape}")
        self.logger.info(f" LABELS: Mean: {torch.mean(y)}, STD: {torch.std(y)} Shape: {y.shape}")

    @staticmethod
    def create_result_str(results, train_loss, valid_loss, epoch):
        result_str = "Epoch: {0}, Train Loss: {1:.4f}, Valid Loss: {2:.4f}, ".format(epoch, train_loss, valid_loss)
        result_str += ', '.join(['{0}: {1:.4f}'.format(k, v.item()) for k, v in results.items() if k != 'Valid Loss'])

        return result_str

    def optimizer_step(self, scores, y):
        loss = F.binary_cross_entropy(scores, y.to(self.device).float())
        loss.backward()
        self.optimizer.step()

        return loss.item()
