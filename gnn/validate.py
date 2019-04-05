import torch
import torch.nn.functional as F
import numpy as np

from gnn.utils import *

import faiss

import logging
logger = logging.getLogger(__name__)


class ClassificationValidator:
    def __init__(self, dev_loader, device):
        self.loader = dev_loader
        self.device = device

    def validate(self, model, verbose=False):
        model.eval()
        accs = []
        losses = []

        for i, (batch_src, batch_tgt, y) in enumerate(self.loader):

            out_src = model(send_to_device(batch_src, self.device))
            out_tgt = model(send_to_device(batch_tgt, self.device))
            y = y.to(self.device)

            scores = torch.sigmoid((out_src * out_tgt).sum(dim=1))
            loss = F.binary_cross_entropy(scores, y.to(self.device).float())

            preds = (scores > 0.5).float()

            if i % 100 == 0 and verbose:
                logger.info(f"VALIDATION - i: {i}")
                logger.info(f" SCORES: Mean: {torch.mean(scores)}, STD: {torch.std(scores)} Shape: {scores.shape}")
                logger.info(
                    f" LABELS: Mean: {torch.mean(y)}, STD: {torch.std(y)} Shape: {y.shape}")
                logger.info(f"SCORES: {scores}")
                logger.info(f"PREDS: {preds}")
                logger.info(f"LABELS: {y}")

            accs.append(torch.eq(preds, y).sum().float().item())
            losses.append(loss.item())

        if verbose:
            logger.info(f"Y    Dev: {y[:50]}")
            logger.info(f"pred Dev: {preds[:50]}")

        avg_acc = torch.mean(torch.tensor(accs)) / self.loader.batch_size
        avg_loss = np.mean(losses)

        return {"Valid Accuracy": avg_acc.numpy(), "Valid Loss": avg_loss}


class RankingValidator:
    def __init__(self, dev_loader, device, measure="l2", src_sents=None):
        self.loader = dev_loader
        self.device = device
        self.measure = measure
        self.src_sents = src_sents

    def validate(self, model, verbose=False):
        model.eval()
        src_embs = []
        tgt_embs = []
        losses = []

        for i, (batch_src, batch_tgt, y) in enumerate(self.loader):
            assert torch.all(y.byte()), f"{y.byte()}"  # for this loader, we should only get positive examples
            src = model(send_to_device(batch_src, self.device))
            tgt = model(send_to_device(batch_tgt, self.device))

            y = y.to(self.device)

            scores = torch.sigmoid((src * tgt).sum(dim=1))
            loss = F.binary_cross_entropy(scores, y.to(self.device).float())

            losses.append(loss.item())

            src_embs.append(src.cpu().data.numpy())
            tgt_embs.append(tgt.cpu().data.numpy())

        src_embs = np.vstack(src_embs)
        tgt_embs = np.vstack(tgt_embs)
        avg_loss = np.mean(losses)

        if self.measure == 'ip':
            index = faiss.IndexFlatIP(src_embs.shape[1])
            logger.info("Using IndexFlatIP")
        else:
            index = faiss.IndexFlatL2(src_embs.shape[1])
            logger.info("Using IndexFlatL2")
        index.add(src_embs)
        _, preds = index.search(tgt_embs.astype(np.float32), 100)
        if not np.any(preds[:, 0]):
            logger.info("ALL PREDICTIONS ARE 0")
            raise RuntimeError

        gold = np.arange(len(tgt_embs))
        result = eval_ranking(preds, gold, [1, 2, 5, 10, 100], verbose=verbose)

        if verbose >= 2:
            logger.info(f"Y    Dev: {y[:50]}")
            logger.info(f"pred Dev: {preds[:50]}")
            check_errors(preds, gold, self.src_sents, [1, 10, 100])

        return {**result, "Valid Loss": avg_loss}
