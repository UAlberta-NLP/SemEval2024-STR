#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse
# private
from src.utils import helper


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # tracks
    parser.add_argument('--track', type=str, default='a', choices=['a', 'b', 'c'])
    # target languages
    # dev for Spanish
    parser.add_argument('--tgt_lan', type=str, default='eng'
        , choices=['eng', 'dev', 'esp', 'spa', 'ind', 'arb', 'arq', 'ary', 'afr', 'amh', 'hau', 'hin', 'kin', 'mar', 'pan', 'tel'])
    # method
    # base for the official baseline method
    parser.add_argument('--method', type=str, default='base', choices=['base'])
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
        self.TRACK_PATH = os.path.join(self.DATA_PATH, self.track)
        self.LAN_PATH = os.path.join(self.TRACK_PATH, self.tgt_lan)
        self.TRAIN_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_train.csv')
        self.DEV_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_dev.csv')
        self.DEV_LABEL_CSV = os.path.join(self.LAN_PATH, f'{self.tgt_lan}_dev_with_labels.csv')
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.track, self.tgt_lan, self.method, str(self.seed)
            )
        os.makedirs(self.LOG_PATH, exist_ok=True)
        self.LOG_TXT = os.path.join(self.LOG_PATH, 'console_log.txt')
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None
        # results
        self.RESULTS_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.track, self.tgt_lan, self.method, str(self.seed)
            )
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        self.RESULTS_CSV = os.path.join(self.RESULTS_PATH, f'pred_{self.tgt_lan}_{self.track}.csv')
        self.RESULTS_ZIP = os.path.join(self.RESULTS_PATH, f'pred_{self.tgt_lan}_{self.track}.csv.zip')