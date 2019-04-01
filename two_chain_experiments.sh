#!/usr/bin/env bash

NUM_SENTS=${1}
BATCH_SIZE=${2}
BPE_VOCABS="500 10000 50000"
WORD_VOCABS="words-lower words-mix"
WORD_SIZES="5000 10000 25000"
DEFAULT="--num_sents ${NUM_SENTS} --batch_size ${BATCH_SIZE} --grid_search"

