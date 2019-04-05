#!/usr/bin/env bash

PARTITION=${1}
NUM_SENTS=${2}
BATCH_SIZE=${3}
NUM_GPUS=${4}

BPE_VOCABS="500 10000 50000"
WORD_VOCABS="words-lower words-mix"
WORD_SIZES="5000 10000 25000"
NEG_SAMPLES="1 2 5 10"
DEFAULT_PARAMS="--num_sents ${NUM_SENTS} --batch_size ${BATCH_SIZE} --grid_search --model_type graph --model gcn --task ranking --grid_search"

if [[ ${PARTITION} == gpu-be ]]
then
    slurm_params="-p gpu-be --exclude eris-1 --cpus-per-task 4 --gres=gpu:${NUM_GPUS} --mem 0 ./train.sh"
else
    slurm_params="-p papago -A papago --exclude eris-1 --cpus-per-task 4 --gres=gpu:${NUM_GPUS} --mem 0 ./train.sh"
fi

for neg_sample in ${NEG_SAMPLES}; do
    echo "Main Chain experiments"
    # Main chain experiments
    for main_vocab in ${BPE_VOCABS}; do
        exp_name="ranking_gcn_main_vocab_${main_vocab}_side_vocabs_None_word_vocab_size_None_neg_samples_${neg_sample}"
        comm="${slurm_params} ${DEFAULT_PARAMS} --main_vocab ${main_vocab} --exp_name ${exp_name} --neg_sample ${neg_sample}"
        echo $comm
        sbatch $comm
    done

    for main_vocab in ${WORD_VOCABS}; do
        for word_size in ${WORD_SIZES}; do
            exp_name="ranking_gcn_main_vocab_${main_vocab}_side_vocabs_None_word_vocab_size_${word_size}_neg_samples_${neg_sample}"
            comm="${slurm_params} ${DEFAULT_PARAMS} --main_vocab ${main_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name}  --neg_sample ${neg_sample}"
            echo $comm
            sbatch $comm
        done
    done


    echo ""
    echo "One side chain experiments:"
    # One side chain experiments
    for main_vocab in ${WORD_VOCABS}; do
        for side_vocab in ${BPE_VOCABS}; do
            for word_size in ${WORD_SIZES}; do
                exp_name="ranking_gcn_main_vocab_${main_vocab}_side_vocabs_${side_vocab}_word_vocab_size_${word_size}_neg_samples_${neg_sample}"
                comm="${slurm_params} ${DEFAULT_PARAMS} --main_vocab ${main_vocab} --side_vocabs ${side_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name} --neg_sample ${neg_sample}"
                echo $comm
                sbatch $comm
            done
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
                        exp_name="ranking_gcn_main_vocab_${main_vocab}_side_vocabs_${side_vocab}_word_vocab_size_${word_size}_neg_samples_${neg_sample}"
                        comm="${slurm_params} ${DEFAULT_PARAMS} --main_vocab ${main_vocab} --side_vocabs ${side_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name} --neg_sample ${neg_sample}"
                        echo $comm
                        sbatch $comm
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
            exp_name="ranking_gcn_main_vocab_${main_vocab}_side_vocabs_${side_vocab}_word_vocab_size_${word_size}_neg_samples_${neg_sample}"
            comm="${slurm_params} ${DEFAULT_PARAMS} --main_vocab ${main_vocab} --side_vocabs ${side_vocab} --word_vocab_size ${word_size} --exp_name ${exp_name} --neg_sample ${neg_sample}"
            echo $comm
            sbatch $comm
        done
    done
done
