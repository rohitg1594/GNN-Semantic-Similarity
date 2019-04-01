import torch
import torch.nn.functional as F
import numpy as np

from gnn.utils import send_to_device

import logging
logger = logging.getLogger(__name__)


class Validator:
    def __init__(self, dev_loader, device):
        self.loader = dev_loader
        self.device = device

    def validate(self, model, verbose=False):
        model.eval()
        accs = []
        losses = []

        for i, (batch_src, batch_tgt, y_dev) in enumerate(self.loader):

            out_src = model(send_to_device(batch_src, self.device))
            out_tgt = model(send_to_device(batch_tgt, self.device))
            y_dev = y_dev.to(self.device)

            scores = torch.sigmoid((out_src * out_tgt).sum(dim=1))
            loss = F.binary_cross_entropy(scores, y_dev.to(self.device).float())

            preds = (scores > 0.5).float()

            if i % 100 == 0 and verbose:
                logger.info(f"VALIDATION - i: {i}")
                logger.info(f" SCORES: Mean: {torch.mean(scores)}, STD: {torch.std(scores)} Shape: {scores.shape}")
                logger.info(
                    f" LABELS: Mean: {torch.mean(y_dev)}, STD: {torch.std(y_dev)} Shape: {y_dev.shape}")
                logger.info(f"SCORES: {scores}")
                logger.info(f"PREDS: {preds}")
                logger.info(f"LABELS: {y_dev}")

            accs.append(torch.eq(preds, y_dev).sum().float().item())
            losses.append(loss.item())

        if verbose:
            logger.info(f"Y    Dev: {y_dev[:50]}")
            logger.info(f"pred Dev: {preds[:50]}")

        return torch.mean(torch.tensor(accs)) / self.loader.batch_size, np.mean(losses)
