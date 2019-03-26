import pickle
import sys
sys.path.append("/nfs/team/nlp/users/rgupta/NMT/code/fairseq/")

from fairseq.gnn.utils import batchify
from fairseq.gnn.models.gcnnet import GCNNet
from fairseq.gnn.models.ginnet import GINNet

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch

import numpy as np

import logging
import argparse
import gc
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--num_sents", type=int, default=10 ** 6)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--exp_name", type=str, default=str(random.randint(0, 10**8)))
parser.add_argument("--dp", type=float, default=0.3)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=1e-05)
parser.add_argument("--train_prop", type=float, default=0.95)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--grid_search", action='store_true', default=False)
parser.add_argument("--verbose", action='store_true', default=False)
parser.add_argument("--model_type", type=str, choices=['gcn', 'gin'], default=False)

args = parser.parse_args()
log_f_name = f"/tmp-network/user/rgupta/logs/semantic_similarity-{args.exp_name}.log"
if os.path.exists(log_f_name):
    os.remove(log_f_name)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(message)s",
    handlers=[
        logging.FileHandler(log_f_name),
        logging.StreamHandler(sys.stdout)
    ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()

str2model = {'gcn': GCNNet,
             'gin': GINNet}

logger.log(20, "Loading dataset")
with open("/nfs/team/nlp/users/rgupta/NMT/code/fairseq/data/local/semantic_similarity.pickle", "rb") as f:
    dataset = pickle.load(f)

logger.log(20, "Dataset loaded")

logger.log(20, "Loading vocab from disk")
with open("/nfs/team/nlp/users/rgupta/NMT/code/fairseq/data/local/semantic_similarity_vocab.pickle", "rb") as f:
    vocab = pickle.load(f)
logger.log(20, "Vocab loaded")

np.random.seed = 42
permute = np.random.permutation(len(dataset)).tolist()

train = []
dev = []
train_size = int(args.train_prop * len(dataset))
dev_size = len(dataset) - train_size

for i in permute:
    if len(train) < train_size:
        train.append(dataset[i])
    else:
        dev.append(dataset[i])
train = train[:args.num_sents]
logger.log(20, f"Num Epochs: {args.num_epochs}, Len dataset: {len(dataset)}, Len train: {len(train)}, Len dev: {len(dev)}, Num Iters: {len(train) // args.batch_size}")

train_data_en = [Data.from_dict({'edge_index': g['edge_index_en'], 'node_ids': g['x_ids_en']}) for g in train]
train_data_de = [Data.from_dict({'edge_index': g['edge_index_de'], 'node_ids': g['x_ids_de']}) for g in train]
train_y = [g['y'] for g in train]

dev_data_en = [Data.from_dict({'edge_index': g['edge_index_en'], 'node_ids': g['x_ids_en']}) for g in dev]
dev_data_de = [Data.from_dict({'edge_index': g['edge_index_de'], 'node_ids': g['x_ids_de']}) for g in dev]
dev_y = [g['y'] for g in dev]


def validate(model, batch_size=32):
    model.eval()
    accs = []

    for i, (batch_en, batch_de, y_dev) in enumerate(zip(batchify(dev_data_en, batch_size),
                                                    batchify(dev_data_de, batch_size),
                                                    batchify(dev_y, batch_size))):
        batch_en = Batch.from_data_list(batch_en).to(device)
        batch_de = Batch.from_data_list(batch_de).to(device)
        y_dev = torch.tensor(y_dev).to(device)

        out_en = model(batch_en)
        out_de = model(batch_de)

        scores = torch.sigmoid((out_en * out_de).sum(dim=1))
        preds = scores > 0.5
        if args.verbose and i % 100 == 0:
            logger.log(20, f"VALIDATION - i: {i}, Mean: {torch.mean(scores)}, STD: {torch.std(scores)}")

        accs.append(torch.eq(preds, y_dev.byte()).sum().item() / batch_size)

    if args.verbose:
        print(f"Y    Dev: {y_dev[:50]}")
        print(f"pred Dev: {preds[:50]}")

    return np.average(accs)


def train(**kwargs):

    lr = kwargs['lr']
    wd = kwargs['wd']
    num_layers = kwargs['num_layers']
    dp = kwargs['dp']
    param2perf = kwargs['param2perf']

    key = f"LR: {lr}, WD: {wd}, NUM_LAYERS: {num_layers}, DP: {dp}"
    logger.log(20, key)
    if param2perf is not None:
        param2perf[key] = []
    model = str2model[args.model_type](num_embs=len(vocab), num_layers=num_layers, dropout=dp).to(device)
    logger.info(f"Model: {model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(args.num_epochs):

        model.train()
        for i, (batch_en, batch_de, y_train) in enumerate(zip(batchify(train_data_en, args.batch_size),
                                                        batchify(train_data_de, args.batch_size),
                                                        batchify(train_y, args.batch_size))):
            batch_en = Batch.from_data_list(batch_en).to(device)
            batch_de = Batch.from_data_list(batch_de).to(device)
            y_train = torch.tensor(y_train, dtype=torch.float).to(device)
            optimizer.zero_grad()
            out_en = model(batch_en)
            out_de = model(batch_de)
            if args.verbose:
                print(f"Batch en: {batch_en.batch.shape}, batch de : {batch_de.batch.shape}, y: {y_train.shape}")
                print(f"Out en: {out_en.shape}, Out de : {out_de.shape}")
            scores = torch.sigmoid((out_en * out_de).sum(dim=1))  # dot product

            if i % 100 == 0:
                logger.log(20, f"TRAIN - i: {i}, Mean: {torch.mean(scores)}, STD: {torch.std(scores)}")

            loss = F.binary_cross_entropy(scores, y_train.float())
            loss.backward()
            optimizer.step()
        acc = validate(model, batch_size=args.batch_size)
        if param2perf is not None:
            param2perf[key].append(acc)
        logger.log(20, f"Epoch: {epoch}, Accuracy: {acc}")

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return param2perf


try:
    if args.grid_search:
            param2perf = {}
            for dp in [0.3, 0.5, 0.7]:
                for lr in [0.01, 0.05, 0.001, 0.005]:
                    for wd in [5e-4, 1e-4, 5e-3, 1e-3, 5e-5, 1e-5]:
                        for num_layers in [2, 3, 4, 5]:

                            param2perf = train(lr=lr, dp=dp, wd=wd, num_layers=num_layers, param2perf=param2perf)

                            logger.log(20, "Saving param2perf disk")
                            with open(f"/nfs/team/nlp/users/rgupta/NMT/code/fairseq/data/local/param2perf-{args.exp_name}.pickle", "wb") as f:
                                pickle.dump(param2perf, f)
                            logger.log(20, "Done")
    else:
        key = f"LR: {args.lr}, WD: {args.wd}, NUM_LAYERS: {args.num_layers}, DP: {args.dp}, EPOCHS: {args.num_epochs}"
        train(lr=args.lr, dp=args.dp, wd=args.wd, num_layers=args.num_layers, param2perf=None)
except Exception as e:
    logger.log(40, "Error", exc_info=1)
