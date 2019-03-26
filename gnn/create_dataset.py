import networkx as nx

from os.path import join
from collections import Counter
import pickle
# import sys
# sys.path.append("/nfs/team/nlp/users/rgupta/NMT/code/fairseq/")

import torch

from gnn.oov_dictionary import OOVDictionary
from gnn.utils import create_graph, rand_exclude


vocab_sizes = [500, 50000]
underscore = "▁"
data_dir = "data"
langs = ['en', 'de']
NUM_SENTS = 10 ** 5
NUM_NEGATIVE = 1

print("reading vocab dictionaries")
dicts = {}
for v in vocab_sizes:
    with open(join(data_dir, f"data/local/train_files/iwslt14-en-de-{v}-no-max-filter-dict.en")) as f:
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
        with open(join(fairseq_dir, f"data/local/train_files/iwslt14-en-de-{v}-no-max-filter-train.{lang}")) as f:
            sents[lang][v] = [line.strip().split() for line in f]
    words[lang] = [''.join(pieces).replace(underscore, ' ')[1:].split() for pieces in sents[lang][vocab_sizes[0]]]

corpus_size = len(words[lang])

print("adding words to shared vocab")
word_counters = {lang: Counter([word for sent in words[lang] for word in sent]) for lang in langs}
word_counts = {lang: [count for word, count in counter.most_common()] for lang, counter in word_counters.items()}
for lang, counter in word_counters.items():
    for word, _ in counter.most_common(25000):
        key = f"word|{word}"
        shared_vocab[key] = len(shared_vocab)

print("saving vocab to disk")
with open("/nfs/team/nlp/users/rgupta/NMT/code/fairseq/data/local/semantic_similarity_vocab.pickle", "wb") as f:
    pickle.dump(shared_vocab, f)

print("creating master dictionary of corpus")
tokens = {}
for lang in langs:
    tokens[lang] = {}
    for v in vocab_sizes:
        tokens[lang][v] = []
        sent_tokens = []
        for i, sent in enumerate(sents[lang][v]):
            if i % 10**3 == 0:
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
        if i % 10 ** 3 == 0:
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

with open("/nfs/team/nlp/users/rgupta/NMT/code/fairseq/data/local/semantic_similarity.pickle", "wb") as f:
    pickle.dump(dataset, f)
