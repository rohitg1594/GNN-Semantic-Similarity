import networkx as nx
from gnn.node import Node
import random
import logging
from torch_geometric.data import Batch
import torch
import sys


def create_graph(head_chain, side_chains, vocab_dict, head_vocab_name='words-lower'):
    """
    Create graph represenation of graph.
    Head chain: List of tokens in main.
    Side chains: Dictionary with key = vocab size and value = list of sentences tokenized with that vocab size.
    vocab_dict: Common vocab dictionary of all nodes in graph.
    """
    g = nx.Graph()
    head_chain_nodes = [Node(f'{head_vocab_name}|{token}', vocab_dict.index(f'{head_vocab_name}|{token}')) for token in head_chain]
    side_chain_nodes = [[[Node(f'{vocab}|{tok}', vocab_dict.index(f'{vocab}|{tok}')) for tok in head] for head in sent]
                                                                        for vocab, sent in side_chains.items()]
    # print(f"Side chain nodes: {side_chain_nodes}")

    for i, word_node in enumerate(head_chain_nodes):
        g.add_node(word_node, label=word_node.name, is_word=True)
        if i > 0:
            g.add_edge(head_chain_nodes[i - 1], head_chain_nodes[i])

    for i, nodes in enumerate(side_chain_nodes):
        for j, head in enumerate(nodes):
            for k, bpe_node in enumerate(head):
                g.add_node(bpe_node, label=bpe_node.name, is_word=False)
                g.add_edge(bpe_node, head_chain_nodes[j])
                if k > 0:
                    g.add_edge(side_chain_nodes[i][j][k], side_chain_nodes[i][j][k - 1])

    return g


def rand_exclude(low, high, size, exclude):
    ans = []
    while len(ans) != size:
        r = random.randint(low, high)
        if r in exclude:
            continue
        else:
            ans.append(r)

    return ans


def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_logger(log_f_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(message)s",
        handlers=[
            logging.FileHandler(log_f_name),
            logging.StreamHandler(sys.stdout)
        ])

    logger = logging.getLogger()

    return logger


def get_logger_2(log_file_name):

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger


def graph_batcher(en_data, de_data, y, device, batch_size=32):
    for i, (en_batch, de_batch, y_batch) in zip(batchify(en_data, batch_size),
                                                batchify(de_data, batch_size),
                                                batchify(y, batch_size)):
        en_batch = Batch.from_data_list(en_batch).to(device)
        de_batch = Batch.from_data_list(de_batch).to(device)
        y_batch = torch.tensor(y, dtype=torch.float).to(device)

        yield en_batch, de_batch, y_batch


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor. Also return len of 1d tensors."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)
    lens = torch.tensor([len(v) - 1 for v in values])

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res, lens


def rand_exclude(low, high, size, exclude):
    ans = []
    while len(ans) != size:
        r = random.randint(low, high)
        if r in exclude:
            continue
        else:
            ans.append(r)

    return ans


def send_to_device(input, device):
    if isinstance(input, list):
        input = [i.to(device) for i in input]
    else:
        input = input.to(device)
    return input


def segment_bpe_sents(sents):
    underscore = "▁"

    res = []
    sent_tokens = []
    for sent in sents:
        buff = [sent[0]]
        for word in sent[1:]:
            if word.startswith(underscore):
                sent_tokens.append(buff)
                buff = [word]
            else:
                buff.append(word)
        sent_tokens.append(buff)  # last buffer
        res.append(sent_tokens)
        sent_tokens = []

    return res
