#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# built-in
import os, argparse
# private
from src.utils import helper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # tracks
    parser.add_argument('--track', type=str, default='a', choices=['a', 'b', 'c', 'd', 'sts'])
    # target languages
    # eng - a, b, c
    # afr - b, c
    # amh - a, b, c
    # arb - b, c
    # arq - a, b, c
    # ary - a, b, c
    # esp - a, b, c
    # hau - a, b, c
    # hin - b, c
    # ind - b, c
    # kin - a, b, c
    # mar - a
    # pan - b, c
    # tel - a, b, c
    parser.add_argument('--tgt_lan', type=str, default='eng'
        , choices= ['eng', 'afr', 'amh', 'arb', 'arq', 'ary', 'esp', 'hau', 'hin', 'ind', 'kin', 'mar', 'pan', 'tel'])
    # method
    # base for the official baseline method
    # sbert
    parser.add_argument('--method', type=str, default='base', choices=['base', 'sbert'])
    # model
    # sentence-transformers/all-mpnet-base-v2
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--max_length', type=int, default=156)
    # train
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    # save as argparse space
    return parser.parse_known_args()[0]


class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        self.update_config(**vars(init_args()))

    def update_config(self, **kwargs):
        # load config from parser
        for k,v in kwargs.items():
            setattr(self, k, v)
        # I/O
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        self.SCORES_PATH = os.path.join(self.RESOURCE_PATH, 'scores')
        self.TRACK_PATH = os.path.join(self.DATA_PATH, self.track)
        self.LAN_PATH = os.path.join(self.TRACK_PATH, self.tgt_lan)
        # data
        self.TRAIN_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_train.csv')
        self.DEV_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_dev.csv')
        self.DEV_LABEL_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_dev_with_labels.csv')
        self.TEST_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_test.csv')
        self.TEST_LABEL_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_test_with_labels.csv')
        # language model
        self.LM_PATH = os.path.join(self.RESOURCE_PATH, 'lm', self.model)
        # checkpoints
        self.CKPT_PATH = os.path.join(
            self.RESOURCE_PATH, 'ckpts', self.track, self.tgt_lan, self.method, self.model, str(self.seed)
            )
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        # results
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.track, self.tgt_lan, self.method, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        self.LOG_TXT = os.path.join(self.LOG_PATH, 'console_log.txt')
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None
        self.RESULTS_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.track, self.tgt_lan, self.method, str(self.seed)
            )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.RESULTS_CSV = os.path.join(self.RESULTS_PATH, f'pred_{self.tgt_lan}_{self.track}.csv')
        self.RESULTS_ZIP = os.path.join(self.RESULTS_PATH, f'pred_{self.tgt_lan}_{self.track}.csv.zip')