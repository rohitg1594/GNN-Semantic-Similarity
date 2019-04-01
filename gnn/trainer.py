import torch
import numpy as np
import torch.nn.functional as F
from gnn.utils import send_to_device


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

    def train(self):

        if self.param2perf is not None:
            key = f"LR: {self.lr}, WD: {self.wd}, NUM_LAYERS: {self.num_layers}, DP: {self.dp}"
            self.logger.info(key)
            key_params = dict([param.strip().split(':') for param in key.split(',')])

        self.logger.info(f"Model: {self.model}")
        self.logger.info("Starting Training....")

        for epoch in range(self.args.num_epochs):

            self.model.train()
            train_losses = []
            for i, (src_batch, tgt_batch, y_batch) in enumerate(self.loader):

                self.optimizer.zero_grad()
                src_batch = send_to_device(src_batch, self.device)
                # print(f"model device: {self.model.get_device()}")
                out_en = self.model(src_batch)
                out_de = self.model(send_to_device(tgt_batch, self.device))
                scores = torch.sigmoid((out_en * out_de).sum(dim=1))  # dot product

                if self.args.verbose:
                    # self.logger.info(f"SOURCE : \n {self.src_dict.string(src_batch[0])}, shape: {src_batch[0].shape}")
                    # self.logger.info(f"TARGET : \n {self.tgt_dict.string(tgt_batch[0])}, shape: {tgt_batch[0].shape}")
                    # self.logger.info(f"Out en: {out_en.shape}, Out de : {out_de.shape}")
                    pass

                if not (((scores >= 0.) & (scores <= 1.)).all() and ((y_batch >= 0.) & (y_batch <= 1.)).all()):
                    raise OverflowError
                else:
                    loss = F.binary_cross_entropy(scores, y_batch.to(self.device).float())
                    train_losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()

                if i % 100 == 0 and self.args.verbose:
                    self.logger.info(f"TRAIN - i: {i}")
                    self.logger.info(f" SCORES: Mean: {torch.mean(scores)}, STD: {torch.std(scores)} Shape: {scores.shape}")
                    self.logger.info(f" LABELS: Mean: {torch.mean(y_batch)}, STD: {torch.std(y_batch)} Shape: {y_batch.shape}")
            acc, valid_loss = self.validator.validate(self.model, verbose=self.args.verbose)
            acc = acc.item()
            train_loss = np.mean(train_losses)
            self.logger.info("Epoch: {0}, Accuracy: {1:.4f}, Train Loss: {2:.4f}, Valid Loss: {3:.4f}".format(epoch,
                                                                                                              acc,
                                                                                                              train_loss,
                                                                                                              valid_loss))
            if self.param2perf is not None:
                res_d = {'Accuracy': acc,
                         'Train Loss': train_loss,
                         'Valid Loss': valid_loss,
                         'Epoch': epoch}
                save_d = {**key_params, **res_d}
                self.param2perf.append(save_d)

        self.logger.info("Training complete.")

        return self.model, self.optimizer, self.param2perf
