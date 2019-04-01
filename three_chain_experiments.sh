#!/usr/bin/env bash

NUM_SENTS=${1}
BATCH_SIZE=${2}
BPE_VOCABS="500 10000 50000"
WORD_VOCABS="words-lower words-mix"
WORD_SIZES="5000 10000 25000"
DEFAULT="--num_sents ${NUM_SENTS} --batch_size ${BATCH_SIZE} --grid_search"

echo ""
echo "Three side chain experiments:"
for main_vocab in ${WORD_VOCABS}; do
    for word_size in ${WORD_SIZES}; do
        side_vocab="500,10000,50000"
        exp_name="model_gcn_main_vocab_${main_vocab}_side_vocabs_${side_vocab}_word_vocab_size_${word_size}"
        comm="${DEFAULT} --main_vocab ${main_vocab} --side_vocabs ${side_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name}"
        echo $comm
        sbatch -p gpu-be --gres=gpu:1 ./train.sh --cpus-per-task 4 --mem 25000 ${comm}
    done
done