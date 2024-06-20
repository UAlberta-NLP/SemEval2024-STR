#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# built-in
import os, random
# public
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer
    , InputExample
    , evaluation
    , losses
    )
# private
from config import Config


MIX = False  # if mix the dev set back with training set
ENG_TRAIN_CSV = f'./res/data/a/eng/eng_train.csv'
ENG_DEV_LABEL_CSV = f'./res/data/a/eng/eng_dev_with_labels.csv'

# initialize the config class
config = Config()
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
for k, v in config.__dict__.items():
    print(f'{k}: {v}')

# Load the pre-trained model
model = SentenceTransformer(config.model)

# Data
# read eng train csv
eng_train_df = pd.read_csv(ENG_TRAIN_CSV)
eng_train_xs1, eng_train_xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in eng_train_df.iterrows()]))
eng_train_ys = eng_train_df['Score'].tolist()
# read eng dev csv
eng_dev_df = pd.read_csv(ENG_DEV_LABEL_CSV)
eng_dev_xs1, eng_dev_xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in eng_dev_df.iterrows()]))
eng_dev_ys = eng_dev_df['Score'].tolist()

# Non-English
if config.tgt_lan != 'eng':
    # read train trans csv
    raw_train_df = pd.read_csv(config.TRAIN_CSV)
    raw_train_ys = raw_train_df['Score'].tolist()
    raw_train_trans_df = pd.read_csv(os.path.join(config.DATA_PATH, 'trans', f'{config.tgt_lan}2eng_train.csv'))
    raw_train_xs1, raw_train_xs2 = map(list, zip(*[(row['Text1 Translation'], row['Text2 Translation']) for idx, row in raw_train_trans_df.iterrows()]))
    # read dev trans csv
    raw_dev_df = pd.read_csv(config.DEV_LABEL_CSV)
    raw_dev_ys = raw_dev_df['Score'].tolist()
    raw_dev_trans_df = pd.read_csv(os.path.join(config.DATA_PATH, 'trans', f'{config.tgt_lan}2eng_dev.csv'))
    raw_dev_xs1, raw_dev_xs2 = map(list, zip(*[(row['Text1 Translation'], row['Text2 Translation']) for idx, row in raw_dev_trans_df.iterrows()]))
    # keep the same dev size
    dev_size = len(raw_dev_ys)
    # MIX
    if MIX:
        raw_train_xs1 += raw_dev_xs1 + eng_train_xs1 + eng_dev_xs1
        raw_train_xs2 += raw_dev_xs2 + eng_train_xs2 + eng_dev_xs2
        raw_train_ys += raw_dev_ys + eng_train_ys + eng_dev_ys
    else:
        raw_train_xs1 +=  eng_train_xs1 + eng_dev_xs1
        raw_train_xs2 +=  eng_train_xs2 + eng_dev_xs2
        raw_train_ys +=  eng_train_ys + eng_dev_ys
# English
else:
    # keep the same dev size
    dev_size = len(eng_dev_ys)
    # MIX
    if MIX:
        raw_train_xs1 = eng_train_xs1 + eng_dev_xs1
        raw_train_xs2 = eng_train_xs2 + eng_dev_xs2
        raw_train_ys = eng_train_ys + eng_dev_ys
    else:
        raw_train_xs1, raw_train_xs2 = eng_train_xs1, eng_train_xs2
        raw_dev_xs1, raw_dev_xs2 = eng_dev_xs1, eng_dev_xs2
        raw_train_ys, raw_dev_ys = eng_train_ys, eng_dev_ys
# data splits
if MIX:
    data_size = len(raw_train_ys)
    train_size = data_size - dev_size
    # shuffle
    idxes = np.arange(data_size).tolist()
    random.shuffle(idxes)
    # split train
    train_idxes = idxes[:train_size]
    train_xs1, train_xs2, train_ys = map(list, zip(*[(x1, x2, y) for i, (x1, x2, y) in enumerate(zip(raw_train_xs1, raw_train_xs2, raw_train_ys)) if i in train_idxes]))
    # split val
    val_xs1, val_xs2, val_ys = map(list, zip(*[(x1, x2, y) for i, (x1, x2, y) in enumerate(zip(raw_train_xs1, raw_train_xs2, raw_train_ys)) if i not in train_idxes])) 
else:
    train_xs1, train_xs2, train_ys = raw_train_xs1, raw_train_xs2, raw_train_ys
    val_xs1, val_xs2, val_ys = raw_dev_xs1, raw_dev_xs2, raw_dev_ys

# information
print('Target Language', config.tgt_lan)
print('If MIX', MIX)
print('Train Size', len(train_xs1), len(train_xs2), len(train_ys))
print('Val Size', len(val_xs1), len(val_xs2), len(val_ys))

# Create InputExamples from the data
train_set = [InputExample(texts=[text1, text2], label=score) for text1, text2, score in zip(train_xs1, train_xs2, train_ys)]

# Define a DataLoader
train_dataloader = DataLoader(
    train_set
    , batch_size=config.train_batch_size
    , shuffle=True
    , pin_memory=True
    , drop_last=True
    )

# Define a loss function, e.g., cosine similarity loss
train_loss = losses.CosineSimilarityLoss(model)

# Create the evaluator
evaluator = evaluation.EmbeddingSimilarityEvaluator(val_xs1, val_xs2, val_ys)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)]
    , epochs=32
    , warmup_steps=128
    , optimizer_class=torch.optim.AdamW
    , optimizer_params={
        'lr': config.learning_rate
        , 'eps': config.adam_epsilon
        }
    , evaluation_steps=512
    , evaluator=evaluator
    , output_path=config.CKPT_PATH
    , save_best_model=True
    , show_progress_bar=True
)
