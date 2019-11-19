from __future__ import print_function

import argparse
import datetime
import json
import os
import numpy

import cupy
import nets as bilm_nets
import chainer
from chainer import training
from chainer.training import extensions
from evaluator import MicroEvaluator

from text_classification import nets as class_nets
from text_classification.nlp_utils import convert_seq
from text_classification import text_datasets

import args_of_text_classifier
from utils import UnkDropout, Outer

class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()

"""load global parameters"""
with open("global.config", "r", encoding='utf-8') as f:
    args = DottableDict(json.load(f))

def main():
    print(json.dumps(args.__dict__, indent=2))
    train(dir="aug_data", print_log=True)

def train(dir="datasets", print_log=False):
    chainer.CHAINER_SEED = args.seed 
    numpy.random.seed(args.seed)

    vocab = None

    """load a dataset"""
    if args.dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            vocab=vocab)
    elif args.dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=args.dataset.endswith('.fine'),
            vocab=vocab)
    elif args.dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                          'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, real_test, vocab = text_datasets.read_text_dataset(
            args.dataset, vocab=None, dir=dir)
    n_class = len(set([int(d[1]) for d in train]))
    
    ## str.format() uses '{}' and ':' to replace '%'
    print(' # train data: {}'.format(len(train)))
    print(' # test  data: {}'.format(len(test)))
    print(' # vocab: {}'.format(len(vocab)))
    print(' # class: {}'.format(n_class))

    chainer.CHAINER_SEED = args.seed 
    numpy.random.seed(args.seed)
    train = UnkDropout(train, vocab['<unk>'], 0.01)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    ## Setup a model
    chainer.CHAINER_SEED = args.seed 
    numpy.random.seed(args.seed)
    if args.model == 'rnn':
        Encoder = class_nets.RNNEncoder
    elif args.model == 'cnn':
        Encoder = class_nets.CNNEncoder
    elif args.model == 'bow':
        Encoder = class_nets.BOWMLPEncoder
    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab),
                      n_units=args.unit, dropout=args.dropout)
    model = class_nets.TextClassifier(encoder, n_class)

    if args.bilm:
        bilm = bilm_nets.BiLanguageModel(
            len(vocab), args.bilm_units, args.bilm_layer, args.bilm_dropout)
        n_labels = len(set([int(v[1]) for v in test]))
        print('# labels = ', n_labels)
        if not args.no_label:
            print('add label')
            bilm.add_label_condition_nets(n_labels, args.bilm_unit)
        else:
            print('not using label')
        chainer.serializers.load_npz(args.bilm, bilm)
        with model.encoder.init_scope():
            initialW = numpy.array(model.encoder.embed.W.data)
            del model.encoder.embed
            model.encoder.embed = bilm_nets.PredictiveEmbed(
                len(vocab), args.unit, bilm, args.dropout,
                initialW=initialW)
            model.encoder.use_predict_embed = True

            model.encoder.embed.setup(
                mode=args.bilm_mode,
                temp=args.bilm_temp,
                word_lower_bound=0.,
                gold_lower_bound=0.,
                gumbel=args.bilm_gumbel,
                residual=args.bilm_residual,
                wordwise=args.bilm_wordwise,
                add_original=args.bilm_add_original,
                augment_ratio=args.bilm_ratio,
                ignore_unk=vocab['<unk>'])
    
    if args.gpu >= 0:
        ## Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu() # copy the model to the GPU
        model.xp.random.seed(args.seed)
    chainer.CHAINER_SEED = args.seed 
    numpy.random.seed(args.seed)

    ## Setup an optimizer
    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)

    ## Setup a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=convert_seq, device=args.gpu)
    
    from triggers import FailMaxValueTrigger
    stop_trigger = FailMaxValueTrigger(
        key='validation/main/accuracy', trigger=(1, 'epoch'),
        n_times=args.stop_epoch, max_trigger=args.epoch)
    trainer = training.Trainer(
        updater, stop_trigger, out=args.out)

    ## Evaluate the model with the test dataset for each epoch
    ## validation set
    trainer.extend(MicroEvaluator(
        test_iter, model,
        converter=convert_seq, device=args.gpu))
    
    if args.validation:
        real_test_iter = chainer.iterators.SerialIterator(
            real_test, args.batchsize,
            repeat=False, shuffle=False)
    eval_on_real_test = MicroEvaluator(
        real_test_iter, model,
        converter=convert_seq, device=args.gpu)
    eval_on_real_test.default_name = 'test'
    trainer.extend(eval_on_real_test)

    ## Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    if args.save_model:
        trainer.extend(extensions.snapshot_object(
            model, 'best_model.npz'),
            trigger=record_trigger)
    
    ## Write a log of evaluation statistics for each epoch
    out = Outer()
    trainer.extend(extensions.LogReport())
    if print_log:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy',
             'test/main/loss', 'test/main/accuracy',
             'elapsed_time']), trigger=record_trigger)
    else:
        trainer.extend(extensions.PrintReport(
            ['main/accuracy', 'validation/main/accuracy',
             'test/main/accuracy'], out=out), trigger=record_trigger)
    
    ## Run the training
    trainer.run()

    ## Free all unused memory blocks "cached" in the memory pool
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
    return float(out[-1])

if __name__ == '__main__':
    main()        

