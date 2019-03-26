import networkx as nx
from fairseq.gnn.node import Node
import random
import logging


def create_graph(words, vocab_dict, other_toks):
    """
    words: List of words in sent.
    other_toks: Dictionary with key = vocab size and value = list of sentences tokenized with that vocab size.
    """
    g = nx.Graph()
    word_nodes = [Node(f'word|{word}', vocab_dict[f'word|{word}']) for word in words]
    bpe_nodes = [[[Node(f'{vocab}|{tok}', vocab_dict[f'{vocab}|{tok}']) for tok in word] for word in sent]
                                                                        for vocab, sent in other_toks.items()]

    for i, word_node in enumerate(word_nodes):
        g.add_node(word_node, label=word_node.name, is_word=True)
        if i > 0:
            g.add_edge(word_nodes[i - 1], word_nodes[i])

    for i, nodes in enumerate(bpe_nodes):
        for j, word in enumerate(nodes):
            for k, bpe_node in enumerate(word):
                g.add_node(bpe_node, label=bpe_node.name, is_word=False)
                g.add_edge(bpe_node, word_nodes[j])
                if k > 0:
                    g.add_edge(bpe_nodes[i][j][k], bpe_nodes[i][j][k - 1])

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


def get_logger(log_file_name):

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger
