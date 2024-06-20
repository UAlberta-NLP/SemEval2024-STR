#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# built-in
import os, sys, random, zipfile, logging, argparse
# public
import numpy as np
# private


def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def init_logger(config):
    # initialize the logger
    file_handler = logging.FileHandler(filename=config.LOG_TXT)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        encoding='utf-8'
        , format='%(asctime)s | %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S'
        , level=logging.INFO
        , handlers=handlers
        )
    logger = logging.getLogger(__name__)
    return logger

def zip_file(file_in, file_out):
    with zipfile.ZipFile(file_out, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(
            file_in
            , arcname=os.path.basename(file_in)
            )

# def get_model(config):
#     match config.method:
#         case 'base':
#             return base.Model()
#         case _:
#             raise NotImplementedError

def seed_everything(seed: int):
    """
    Seed everything to ensure reproducibility.
    
    Parameters:
    seed (int): The seed value to use for seeding.
    """
    random.seed(seed)
    np.random.seed(seed)