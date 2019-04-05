import pickle
from os.path import join

from gnn.utils import get_logger
from gnn.models.bilstm import BILSTM
from gnn.models.gcnnet import GCNNet
from gnn.models.ginnet import GINNet

from gnn.data.plain_sent import PlainSentDataset
from gnn.data.dictionary import Dictionary
from gnn.data.graph_sent import GraphSentDataset

from gnn.trainer import Trainer
from gnn.validate import ClassificationValidator, RankingValidator

import torch
import torch.nn as nn

import argparse
import gc
import os
import random

try:

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--task", type=str, choices=['ranking', 'classification'], default='classification')
    parser.add_argument("--measure", type=str, choices=['l2', 'ip'], default='ip', help='if ranking task, distance measure for retrieval')

    parser.add_argument("--num_sents", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="iwslt14")
    parser.add_argument("--src_lang", type=str, default="en")
    parser.add_argument("--tgt_lang", type=str, default="de")
    parser.add_argument("--save_graphs", action='store_true')
    parser.add_argument("--exp_name", type=str, default="random/" + str(random.randint(0, 10**8)))
    parser.add_argument("--neg_sample", type=int, default=1)
    parser.add_argument("--grid_search", action='store_true')
    parser.add_argument("--model_type", type=str, choices=['graph', 'plain'], default='graph')
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--side_vocabs", type=str, help="comma separated list of side vocabs to use for representation")
    parser.add_argument("--main_vocab", type=str, help="if using graph model type, then vocab name of main chain", default=10000)
    parser.add_argument("--word_vocab_size", type=int, help="if vocab option is words, then choose size of vocab", default=-1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dp", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-05)
    parser.add_argument("--optimizer", type=str, choices=['adam'], default='adam')

    parser.add_argument("--model", type=str, choices=['bilstm', 'gcn', 'gin'], default='gcn')
    parser.add_argument("--aggr", type=str, choices=['mean', 'max', 'sum'], default='mean')
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()
    assert args.data_dir is not None
    assert args.log_dir is not None
    assert args.main_vocab is not None
    if args.model_type == "graph":
        assert args.model != "bilstm", "If model type is graph, then model should be graph based and not bilstm"
    if args.model_type == "plain":
        assert args.model == "bilstm", "If model type is graph, then model should be bilsmt  and not graph based"
        assert args.side_vocabs is None

    all_vocabs = [args.main_vocab]
    if args.side_vocabs is not None:
        all_vocabs += args.side_vocabs.split(",")

    log_f_name = join(args.log_dir, f"{args.exp_name}.log")
    if os.path.exists(log_f_name):
        os.remove(log_f_name)
    logger = get_logger(log_f_name)
    logger.info("Experiment Parameters")
    for arg in sorted(vars(args)):
        logger.info('{:<15}\t{}'.format(arg, getattr(args, arg)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    MODEL_REGISTRY = {'bilstm': BILSTM,
                      'gcn': GCNNet,
                      'gin': GINNet
                      }

    if args.model_type == 'plain':
        src_dict = Dictionary.load(join(args.data_dir, f'vocabs/{args.dataset}/{args.main_vocab}.{args.src_lang}'), prefix="")
        src_dict.finalize(nwords=args.word_vocab_size)
        tgt_dict = Dictionary.load(join(args.data_dir, f'vocabs/{args.dataset}/{args.main_vocab}.{args.tgt_lang}'), prefix="")
        tgt_dict.finalize(nwords=args.word_vocab_size)
    else:
        src_vocab2dict = {vocab: Dictionary.load(join(args.data_dir, f"vocabs/iwslt14/{vocab}.{args.src_lang}"), prefix=vocab) for vocab in all_vocabs}
        tgt_vocab2dict = {vocab: Dictionary.load(join(args.data_dir, f"vocabs/iwslt14/{vocab}.{args.tgt_lang}"), prefix=vocab) for vocab in all_vocabs}

        vocab2lens = {vocab: args.word_vocab_size if 'words' in str(vocab) else -1 for vocab in all_vocabs}
        src_dict = Dictionary.merge_dictionaries([(src_vocab2dict[v], vocab2lens[v]) for v in all_vocabs])
        tgt_dict = Dictionary.merge_dictionaries([(tgt_vocab2dict[v], vocab2lens[v]) for v in all_vocabs])

    logger.info(f"Dictionaries loaded, len src: {len(src_dict)}, len tgt: {len(tgt_dict)}")
    combined_dictionary = Dictionary.merge_dictionaries([(src_dict, -1), (tgt_dict, -1)])
    logger.info(f"Created combined dictionary, len combined: {len(combined_dictionary)}")
    logger.info(f"Dummy sentence: {combined_dictionary.string(combined_dictionary.dummy_sentence(10))}")
    del src_dict, tgt_dict

    splits = ['train', 'valid']  # , 'test']
    datasets = {}
    loaders = {}
    for split in splits:
        if args.task == 'ranking' and split == 'valid':
            logger.info('Setting neg sample to zero')
            neg_sample = 0
        else:
            neg_sample = args.neg_sample

        if args.model_type == 'plain':
            src_train_f = join(args.data_dir, f'corpora/{args.dataset}/{args.main_vocab}/{split}.{args.src_lang}')
            tgt_train_f = join(args.data_dir, f'corpora/{args.dataset}/{args.main_vocab}/{split}.{args.tgt_lang}')

            dataset = PlainSentDataset(src_data_fname=src_train_f,
                                       tgt_data_fname=tgt_train_f,
                                       dictionary=combined_dictionary,
                                       logger=logger,
                                       num_sents=args.num_sents,
                                       neg_sample=neg_sample,
                                       prefix="",
                                       verbose=args.verbose)
        else:
            src_vocab2data_fname = {vocab: join(args.data_dir, f"corpora/{args.dataset}/{vocab}/{split}.{args.src_lang}") for vocab in all_vocabs}
            tgt_vocab2data_fname = {vocab: join(args.data_dir, f"corpora/{args.dataset}/{vocab}/{split}.{args.tgt_lang}") for vocab in all_vocabs}

            dataset = GraphSentDataset(src_main_vocab=args.main_vocab,
                                       src_vocab2data_fname=src_vocab2data_fname,
                                       tgt_main_vocab=args.main_vocab,
                                       tgt_vocab2data_fname=tgt_vocab2data_fname,
                                       dictionary=combined_dictionary,
                                       verbose=args.verbose,
                                       logger=logger,
                                       num_sents=args.num_sents,
                                       args=args,
                                       neg_sample=neg_sample)

        loader = dataset.get_loader(batch_size=args.batch_size)

        datasets[split] = dataset
        loaders[split] = loader

    len_train = len(datasets['train'])
    len_valid = len(datasets['valid'])
    logger.info(
        f"Num Epochs: {args.num_epochs}, Len train: {len_train}, Len Valid: {len_valid}, Num Iters: {len_train // args.batch_size}")

    if args.task == 'classification':
        validator = ClassificationValidator(loaders['valid'], device)
    elif args.task == 'ranking':
        validator = RankingValidator(loaders['valid'], device, measure=args.measure, src_sents=datasets['valid'].src_raw)
    logger.info("Created validator.")

    def get_model(input_size=args.input_size,
                  hidden_size=args.hidden_size,
                  num_embs=len(combined_dictionary),
                  num_layers=args.num_layers,
                  dropout=args.dp,
                  output_size=args.output_size,
                  aggr=args.aggr):
        model = MODEL_REGISTRY[args.model]
        logger.info(f"Model: {model}")
        model = model(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=output_size,
                      num_embs=num_embs,
                      num_layers=num_layers,
                      dropout=dropout,
                      aggr=aggr)

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs.")
            model = nn.DataParallel(model)
        model = model.to(device)

        return model

    def get_trainer(lr=args.lr,
                    param2perf=None,
                    wd=args.wd,
                    num_layers=args.num_layers,
                    dp=args.dp,
                    logger=logger,
                    model=None,
                    device=device,
                    loader=loaders['train'],
                    args=args,
                    validator=validator,
                    optimizer=args.optimizer,
                    dictionary=combined_dictionary,
                    ):

        assert model is not None
        return Trainer(lr=lr,
                       param2perf=param2perf,
                       wd=wd,
                       num_layers=num_layers,
                       dp=dp,
                       logger=logger,
                       model=model,
                       device=device,
                       loader=loader,
                       args=args,
                       validator=validator,
                       optimizer=optimizer,
                       dictionary=dictionary,)

    if args.grid_search:
        exp_params = {"word_vocab_size": args.word_vocab_size,
                      "main_vocab": args.main_vocab,
                      "side_vocabs": args.side_vocabs,
                      "model": args.model}
        param2perf = []
        for dp in [0.3, 0.5]:
            for wd in [5e-5, 1e-5, 5e-4, 1e-4]:
                for lr in [0.001, 0.005, 0.01,]:
                    model = get_model(dropout=dp)
                    trainer = get_trainer(lr=lr,
                                          param2perf=param2perf,
                                          wd=wd,
                                          dp=dp,
                                          model=model)
                    try:
                        model, optimzer, param2perf = trainer.train()
                    except OverflowError:
                        logger.error("Overflow error, continuing with next settings")
                        continue

                    del model, optimzer
                    gc.collect()

                    logger.info("Saving param2perf 2 disk")

                    print(f"Param2perf len: {len(param2perf)}")
                    param2perf = [{**exp_params, **record} for record in param2perf]
                    for record in param2perf:
                        print(record)
                    with open(join(args.data_dir, f"grid_search/{args.exp_name}.pkl"), "wb") as f:
                        pickle.dump(param2perf, f)
                    logger.info("Done")

    else:
        logger.info(f"LR: {args.lr}, WD: {args.wd}, NUM_LAYERS: {args.num_layers}, DP: {args.dp}, EPOCHS: {args.num_epochs}")
        model = get_model()
        trainer = get_trainer(model=model)

        logger.info("Created training")
        logger.info("Launching training...")
        trainer.train()

except Exception as e:
    logger.error("Error", exc_info=1)
