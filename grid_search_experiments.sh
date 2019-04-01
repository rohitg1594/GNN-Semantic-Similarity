#!/usr/bin/env bash

NUM_SENTS=${1}
BATCH_SIZE=${2}
BPE_VOCABS="500 10000 50000"
WORD_VOCABS="words-lower words-mix"
WORD_SIZES="5000 10000 25000"
DEFAULT="--num_sents ${NUM_SENTS} --batch_size ${BATCH_SIZE} --grid_search"

echo "Main Chain experiments"
# Main chain experiments
for main_vocab in ${BPE_VOCABS}; do
    exp_name="model_gcn_main_vocab_${main_vocab}_side_vocabs_None_word_vocab_size_None"
    comm="${DEFAULT} --main_vocab ${main_vocab} --exp_name ${exp_name}"
    sbatch -p papago -A papago --exclude eris-1 --gres=gpu:1 --cpus-per-task 4 --mem 25000 ./train.sh ${comm}
done

for main_vocab in ${WORD_VOCABS}; do
    for word_size in ${WORD_SIZES}; do
        exp_name="model_gcn_main_vocab_${main_vocab}_side_vocabs_None_word_vocab_size_${word_size}"
        comm="${DEFAULT} --main_vocab ${main_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name}"
        echo $comm
        sbatch -p papago -A papago --exclude eris-1 --gres=gpu:1 --cpus-per-task 4 --mem 25000 ./train.sh ${comm}
    done
done


echo ""
echo "Two side chain experiments:"
# One side chain experiments
for main_vocab in ${WORD_VOCABS}; do
    for side_vocab1 in ${BPE_VOCABS}; do
        for side_vocab2 in ${BPE_VOCABS}; do
            for word_size in ${WORD_SIZES}; do
                if [ ${side_vocab1} -ne ${side_vocab2} ]; then
                    side_vocab="${side_vocab1},${side_vocab2}"
                    exp_name="model_gcn_main_vocab_${main_vocab}_side_vocabs_${side_vocab}_word_vocab_size_${word_size}"
                    comm="${DEFAULT} --main_vocab ${main_vocab} --side_vocabs ${side_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name}"
                    echo $comm
                    sbatch -p papago -A papago --exclude eris-1 --gres=gpu:1 --cpus-per-task 4 --mem 25000 ./train.sh ${comm}
                fi
            done
        done
    done
done


echo ""
echo "Three side chain experiments:"
for main_vocab in ${WORD_VOCABS}; do
    for word_size in ${WORD_SIZES}; do
        side_vocab="500,10000,50000"
        exp_name="model_gcn_main_vocab_${main_vocab}_side_vocabs_${side_vocab}_word_vocab_size_${word_size}"
        comm="${DEFAULT} --main_vocab ${main_vocab} --side_vocabs ${side_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name}"
        echo $comm
        sbatch -p papago -A papago --exclude eris-1 --gres=gpu:1 --cpus-per-task 4 --mem 25000 ./train.sh ${comm}
    done
done



