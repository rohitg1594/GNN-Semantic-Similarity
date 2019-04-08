from gnn.node import Node

import random
import logging
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from torch_geometric.data import Batch
import torch


def plot_graph(g, f_name):
    fig = plt.figure(figsize=(20, 20))
    print(f"nodes: {[node for node in g.nodes()]}")
    print(f"nodes: {[node.is_main for node in g.nodes()]}")
    #groups = set(nx.get_node_attributes(g, 'is_main').values())
    #mapping = {k: 1 if k else 0 for k in groups}
    #print(f"mapping:{mapping}, groups: {groups}")
    nodes = g.nodes()
    colors = [1 if n.is_main else 0 for n in g.nodes()]

    pos = nx.kamada_kawai_layout(g)
    ec = nx.draw_networkx_edges(g, pos, alpha=0.5)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes,
                                with_labels=True, node_size=200, cmap=plt.cm.Pastel1,
                                node_color=colors,
                                font_weight='bold')
    labels = nx.draw_networkx_labels(g, pos)
    plt.savefig(f_name)


def create_graph(main_chain, side_chains, dictionary, main_vocab='words-lower',):
    """
    Create graph represenation of graph.
    Head chain: List of tokens in main.
    Side chains: Dictionary with key = vocab size and value = list of sentences tokenized with that vocab size.
    vocab_dict: Common vocab dictionary of all nodes in graph.
    """
    g = nx.Graph()
    # print(f"Symbols: {dictionary.symbols[:100]}")
    # print(f"Items: {list(dictionary.indices.items())[:100]}")
    # print(f"Main chain: {main_chain}")
    # for token in main_chain:
    #    pr_token = token
    #    if pr_token in dictionary.indices:
    #        print(f"{pr_token} is in dictionary")
    #    else:
    #        print(f"{pr_token} is not in dictionary")
    # print(f"Main chain ids: {[dictionary.index(token) for token in main_chain]}")
    # exit()
    main_chain_nodes = [Node(token, dictionary.index(token), 1) for token in main_chain]
    side_chain_nodes = [[[Node(token, dictionary.index(token), 0) for token in head] for head in sent] for vocab, sent in side_chains.items()]
    # print(f"Side chain nodes: {side_chain_nodes}")
    # print(f"Main chain nodes: {main_chain_nodes}")

    for i, word_node in enumerate(main_chain_nodes):
        g.add_node(word_node, label=word_node.name, is_word=True)
        if i > 0:
            g.add_edge(main_chain_nodes[i - 1], main_chain_nodes[i])

    for i, nodes in enumerate(side_chain_nodes):
        for j, head in enumerate(nodes):
            for k, bpe_node in enumerate(head):
                g.add_node(bpe_node, label=bpe_node.name, is_word=False)
                g.add_edge(bpe_node, main_chain_nodes[j])
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


def segment_bpe_sents(sents, eos):
    underscore = "▁"

    res = []
    sent_tokens = []
    for sent in sents:
        buff = [sent[0]]
        for word in sent[1:]:
            if word.startswith(underscore) or word == eos:
                sent_tokens.append(buff)
                buff = [word]
            else:
                buff.append(word)
        sent_tokens.append(buff)  # last buffer
        res.append(sent_tokens)
        sent_tokens = []

    return res


def equal_length(items):
    return all(len(x) == len(items[0]) for x in items)


def check_errors(I, gold, src_sents, ks):
    errors = defaultdict(list)

    for j, k in enumerate(ks):
        for i in range(I.shape[0]):
            if gold[i] not in I[i, :k]:
                errors[k].append((gold[i], I[i]))

    for k, errors in errors.items():
        print("Top {} errors:".format(k))
        mask = random.sample(range(len(errors)), 10)
        for i in mask:
            gold_id, predictions_id = errors[i]

            print('{}|{}'.format(src_sents[gold_id], ';'.join([src_sents[id] for id in predictions_id])))
    print()


def eval_ranking(I, gold, ks, verbose=False):
    assert len(I) == len(gold), f"I: {I.shape}, gold: {gold.shape}"
    if verbose:
        print(f"gold: {gold[:10]}, I: {I[:10, :20]}")
    out = {k: 0 for k in ks}

    for k in ks:
        for i in range(I.shape[0]):
            if gold[i] in I[i, :k]:
                out[k] += 1

    out = {k: np.array(v / I.shape[0]) for k, v in out.items()}

    # Mean Reciprocal Rank
    ranks = []
    for i in range(I.shape[0]):
        index = np.where(gold[i] == I[i])[0] + 1
        if not index:
            ranks.append(1 / I.shape[1])
        else:
            ranks.append(1 / index)
    mrr = np.mean(np.array(ranks))

    if not isinstance(mrr, float):
        mrr = mrr[0]

    out['mrr'] = mrr

    return out


def create_example_ids(num_pos, neg_sample=1):
    pos_ids = [(i, i) for i in range(num_pos)]

    if neg_sample:
        neg_ids = [(i, j) for i in range(num_pos) for j in rand_exclude(0, num_pos - 1, neg_sample, [i])]
        y = torch.cat((torch.ones(num_pos), torch.zeros(neg_sample * num_pos)))
    else:
        neg_ids = []
        y = torch.ones(num_pos)

    all_ids = pos_ids + neg_ids

    return pos_ids, neg_ids, all_ids, y
