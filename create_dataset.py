import networkx as nx

from os.path import join
from collections import Counter
import pickle
import argparse
import sys

import torch

import numpy as np

from gnn.oov_dictionary import OOVDictionary
from gnn.utils import create_graph, rand_exclude

parser = argparse.ArgumentParser()
parser.add_argument("--num_sents", type=int, default=10 ** 6)
parser.add_argument("--train_prop", type=float, default=0.9)
args = parser.parse_args()


vocab_sizes = [500, 50000]
underscore = "▁"
data_dir = "data"
langs = ['en', 'de']
NUM_SENTS = 10 ** 6
NUM_NEGATIVE = 1
WORD_THRESHOLD = 25000

print("reading vocab dictionaries")
dicts = {}
for v in vocab_sizes:
    with open(join(data_dir, f"vocabs/iwslt14-en-de-{v}-vocab.txt")) as f:
        dicts[v] = dict([line.strip().split() for line in f])

print("creating shared Vocab Dictionary")
shared_vocab = {'OOV': 0}
for v in vocab_sizes:
    for token in dicts[v].keys():
        k = f"{v}|{token}"
        shared_vocab[k] = len(shared_vocab)

print("reading corpus tokenized with different vocabs")
sents = {}
words = {}
for lang in langs:
    sents[lang] = {}
    for v in vocab_sizes:
        with open(join(data_dir, f"corpora/iwslt14-en-de-{v}-train.{lang}")) as f:
            sents[lang][v] = [line.strip().split() for line in f]
    words[lang] = [''.join(pieces).replace(underscore, ' ')[1:].split() for pieces in sents[lang][vocab_sizes[0]]]

corpus_size = len(words[lang])

print("adding words to shared vocab")
word_counters = {lang: Counter([word for sent in words[lang] for word in sent]) for lang in langs}
word_counts = {lang: [count for word, count in counter.most_common()] for lang, counter in word_counters.items()}
for lang, counter in word_counters.items():
    for word, _ in counter.most_common(WORD_THRESHOLD):
        key = f"word|{word}"
        shared_vocab[key] = len(shared_vocab)

print("saving word counters")
with open(join(data_dir, "vocabs/word_counters.pkl"), "wb") as f:
    pickle.dump(word_counters, f)

print("saving vocab to disk")
with open(join(data_dir, "vocabs/bpe-500-50000-word-thresh-25000-vocab.pkl"), "wb") as f:
    pickle.dump(shared_vocab, f)

print("creating master dictionary of corpus")
tokens = {}
for lang in langs:
    tokens[lang] = {}
    for v in vocab_sizes:
        tokens[lang][v] = []
        sent_tokens = []
        for i, sent in enumerate(sents[lang][v]):
            if i % 10 ** 4 == 0:
                print(lang, v, i)
            buff = [sent[0]]
            for word in sent[1:]:
                if word.startswith(underscore):
                    sent_tokens.append(buff)
                    buff = [word]
                else:
                    buff.append(word)
            sent_tokens.append(buff)  # last buffer
            tokens[lang][v].append(sent_tokens)
            sent_tokens = []

print("saving tokens dataset")
with open(join(data_dir, "processed/tokens_dataset.pkl"), "wb") as f:
    pickle.dump(tokens, f)

# Check different vocab sizes have same sentence length
for lang in langs:
    for v in vocab_sizes:
        for i in range(corpus_size):
            assert len(words[lang][i]) == len(tokens[lang][v][i]), (lang, v, i, len(words[lang][i]), len(tokens[lang][v][i]))

vocab_dict = OOVDictionary(shared_vocab)

print("creating graphs ids and edges dicts")
graph_ids = {}
graph_edges = {}
for lang in langs:
    graph_ids[lang] = []
    graph_edges[lang] = []
    for i in range(corpus_size):
        if i % 10 ** 4 == 0:
            print(lang, i)
        g = create_graph(words[lang][i], vocab_dict, {vocab: tokens[lang][vocab][i] for vocab in vocab_sizes})
        int_g = nx.convert_node_labels_to_integers(g)
        edges = torch.tensor([e for e in int_g.edges], dtype=torch.long).t().contiguous()

        graph_ids[lang].append(torch.tensor([node.id for node in g]))
        graph_edges[lang].append(edges)

del tokens

print("creating dataset")
# True matches
dataset = [{'x_ids_en': graph_ids['en'][i],
            'x_ids_de': graph_ids['de'][i],
            'edge_index_en': graph_edges['en'][i],
            'edge_index_de': graph_edges['de'][i],
            'y': 1} for i in range(corpus_size)]

# Negative sampling
dataset = dataset + [{'x_ids_en': graph_ids['en'][i],
                      'x_ids_de': graph_ids['de'][j],
                      'edge_index_en': graph_edges['en'][i],
                      'edge_index_de': graph_edges['de'][j],
                      'y': 0} for i in range(corpus_size) for j in rand_exclude(0, corpus_size - 1, NUM_NEGATIVE, [i])]


np.random.seed = 42
permute = np.random.permutation(len(dataset)).tolist()

train = []
dev = []
test = []
train_size = int(args.train_prop * len(dataset))
dev_size = test_size = (len(dataset) - train_size) // 2

for i in permute:
    if len(train) < train_size:
        train.append(dataset[i])
    elif len(dev) < dev_size:
        dev.append(dataset[i])
    else:
        test.append(dataset[i])

print(f"saving graphs dataset, train: {len(train)}, dev: {len(dev)}, test: {len(test)}")
with open(join(data_dir, "processed/graphs_dataset.pkl"), "wb") as f:
    pickle.dump({'train': train, 'dev': dev, 'test': test}, f)
