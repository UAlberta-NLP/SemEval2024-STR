#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

"""
Paraphrase Identification (PI) Method for Semantic Textual Relatedness (STR)

This module implements a two-phase approach:
  Phase 1 (Training): Train RoBERTa binary classifier on general paraphrase datasets
    - Datasets: PIT, QQP, MRPC, PAWS QQP, PAWS Wiki, PARADE
    - Standalone training: no STR task data mixed in
    - Outputs checkpoint to config.CKPT_PATH

  Phase 2 (Inference): Apply trained model to SemEval STR task
    - Loads checkpoint from config.CKPT_PATH
    - Bidirectional probability averaging: score = avg(P(paraphrase|x1,x2), P(paraphrase|x2,x1))
    - Outputs predictions to config.RESULTS_CSV

Usage:
  python pi.py --track a --tgt_lan eng --seed 0
    - Trains model on paraphrase data (Phase 1) → Infers on STR task (Phase 2)
    - Results saved to res/results/a/eng/pi/0/pred_eng_a.csv
"""


# dependency
# built-in
import os, sys, random, argparse
# public
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
)
import lightning.pytorch as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
# private
from config import Config
from src.utils import helper


# Training configuration
MODEL_NAME = 'roberta-base'
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 32
EARLY_STOPPING_PATIENCE = 3
SEED = 42


class PIClassifier(pl.LightningModule):
    """
    RoBERTa-based binary classifier for paraphrase identification.

    Inherits from PyTorch Lightning for training automation. Outputs probability
    of paraphrase for text pairs, used as relatedness score in STR task.

    Args:
        model_name: HuggingFace model identifier (default: roberta-base)
        learning_rate: Adam learning rate (default: 5e-5)
        weight_decay: L2 regularization (default: 0.01)
        adam_epsilon: Adam epsilon for numerical stability (default: 1e-8)
    """
    def __init__(self, model_name=MODEL_NAME, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, adam_epsilon=ADAM_EPSILON):
        super(PIClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: 'negative', 1: 'positive'},
            label2id={'negative': 0, 'positive': 1},
            ignore_mismatched_sizes=True,
        )
        self.train_loss_list = []
        self.train_acc, self.train_f1 = BinaryAccuracy(), BinaryF1Score()
        self.val_acc, self.val_f1 = BinaryAccuracy(), BinaryF1Score()
        self.val_pos_f1 = BinaryF1Score(ignore_index=0)
        self.test_acc, self.test_f1 = BinaryAccuracy(), BinaryF1Score()
        self.test_pos_f1 = BinaryF1Score(ignore_index=0)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        outputs = self.model(**xs, labels=ys)
        loss = outputs.loss.item()
        self.train_loss_list.append(loss)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        self.log('train_step_loss', loss, prog_bar=True)
        self.train_acc.update(ys_, ys)
        self.train_f1.update(ys_, ys)
        return outputs.loss

    def on_train_epoch_end(self):
        self.log_dict({
            'train_epoch_loss': np.mean(self.train_loss_list, dtype='float32'),
            'train_acc': self.train_acc.compute(),
            'train_f1': self.train_f1.compute(),
        })
        self.train_loss_list = []
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        _, batch = batch
        xs, ys = batch
        outputs = self.model(**xs, labels=None)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        self.val_acc.update(ys_, ys)
        self.val_f1.update(ys_, ys)
        self.val_pos_f1.update(ys_, ys)

    def on_validation_epoch_end(self):
        self.log_dict({
            'val_acc': self.val_acc.compute(),
            'val_f1': self.val_f1.compute(),
            'val_pos_f1': self.val_pos_f1.compute(),
        })
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_pos_f1.reset()

    def test_step(self, batch, batch_idx):
        _, batch = batch
        xs, ys = batch
        ys_ = self.model(**xs, labels=None).logits.softmax(dim=1).argmax(dim=1)
        self.test_acc.update(ys_, ys)
        self.test_f1.update(ys_, ys)
        self.test_pos_f1.update(ys_, ys)

    def on_test_epoch_end(self):
        self.log_dict({
            'test_acc': self.test_acc.compute(),
            'test_f1': self.test_f1.compute(),
            'test_pos_f1': self.test_pos_f1.compute(),
        })
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_pos_f1.reset()

    def predict_step(self, batch, batch_idx):
        (xs0, xs1, ys), (xs, _) = batch
        outputs = self.model(**xs, labels=None)
        ys_ = outputs.logits.softmax(dim=1).argmax(dim=1)
        return {'xs0': xs0, 'xs1': xs1, 'ys': ys, 'ys_': ys_}

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )
        return optimizer

    def forward(self, **kwargs):
        return self.model(**kwargs)


class PIDataset(Dataset):
    """
    PyTorch Dataset for paraphrase identification with text pairs.

    Args:
        texts1: List of first texts
        texts2: List of second texts
        labels: List of binary labels (0/1 for non-paraphrase/paraphrase)
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length for tokenization (default: 128)
    """
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=128):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text1,
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader. Stacks batch items into tensors."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }, labels


def train_pi_model(config):
    """
    Phase 1: Train RoBERTa binary classifier on paraphrase datasets.

    Loads paraphrase data from res/data/paraphrase/ (PIT, QQP, MRPC, PAWS, PARADE),
    performs 80/20 train/val split, and trains with early stopping.
    Checkpoint saved to config.CKPT_PATH/lightning_logs/version_0/checkpoints/

    Args:
        config: Config instance with CKPT_PATH and seed

    Returns:
        Trained PIClassifier model or None if datasets not found
    """
    print('\n' + '=' * 50)
    print('Phase 1: Training PI Model')
    print('=' * 50)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(f'Model: {MODEL_NAME}')
    print(f'Max Length: {MAX_LENGTH}')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Learning Rate: {LEARNING_RATE}')
    print(f'Max Epochs: {MAX_EPOCHS}')
    print(f'Checkpoint Path: {config.CKPT_PATH}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print('\nLoading paraphrase datasets...')
    datasets_paths = [
        './res/data/paraphrase/pit_train.csv',
        './res/data/paraphrase/qqp_train.csv',
        './res/data/paraphrase/mrpc_train.csv',
        './res/data/paraphrase/paws_qqp_train.csv',
        './res/data/paraphrase/paws_wiki_train.csv',
        './res/data/paraphrase/parade_train.csv',
    ]

    all_texts1, all_texts2, all_labels = [], [], []
    for path in datasets_paths:
        if os.path.exists(path):
            print(f'  Loading {path}...')
            df = pd.read_csv(path)
            all_texts1.extend(df['text1'].tolist())
            all_texts2.extend(df['text2'].tolist())
            all_labels.extend(df['label'].tolist())
        else:
            print(f'  Warning: {path} not found, skipping...')

    if not all_texts1:
        print('Error: No paraphrase datasets found!')
        print('Please download paraphrase datasets to res/data/paraphrase/')
        return None

    print(f'Total samples loaded: {len(all_texts1)}')

    data_size = len(all_labels)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    split_idx = int(data_size * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_texts1 = [all_texts1[i] for i in train_indices]
    train_texts2 = [all_texts2[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]

    val_texts1 = [all_texts1[i] for i in val_indices]
    val_texts2 = [all_texts2[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    print(f'Train Size: {len(train_texts1)}')
    print(f'Val Size: {len(val_texts1)}')

    train_dataset = PIDataset(train_texts1, train_texts2, train_labels, tokenizer, max_length=MAX_LENGTH)
    val_dataset = PIDataset(val_texts1, val_texts2, val_labels, tokenizer, max_length=MAX_LENGTH)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: (
            {
                'input_ids': torch.stack([b['input_ids'] for b in batch]),
                'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            },
            torch.stack([b['labels'] for b in batch]),
        ),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: (
            {
                'input_ids': torch.stack([b['input_ids'] for b in batch]),
                'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            },
            torch.stack([b['labels'] for b in batch]),
        ),
    )

    print('\nInitializing model...')
    model = PIClassifier(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON,
    )

    print('Starting training...')
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        log_every_n_steps=50,
        default_root_dir=config.CKPT_PATH,
        enable_checkpointing=True,
        callbacks=[EarlyStoppingCallback(monitor='val_acc', patience=EARLY_STOPPING_PATIENCE, mode='max')],
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    print(f'\nTraining completed! Model saved to: {config.CKPT_PATH}')
    return model


def infer_pi_model(config, model=None):
    """
    Phase 2: Inference on SemEval STR task using bidirectional probability averaging.

    Loads trained checkpoint and computes paraphrase probability for each text pair.
    Uses bidirectional scoring to reduce directional bias:
      score = avg(P(paraphrase|text1,text2), P(paraphrase|text2,text1))

    Outputs predictions to config.RESULTS_CSV with format: PairID, Pred_Score

    Args:
        config: Config instance with DEV_LABEL_CSV, RESULTS_CSV paths
        model: Optional pre-trained PIClassifier. If None, loads from config.CKPT_PATH

    Returns:
        DataFrame with predictions
    """
    print('\n' + '=' * 50)
    print('Phase 2: Inference on STR Task Data')
    print('=' * 50)

    if model is None:
        print(f'Loading model from checkpoint: {config.CKPT_PATH}')
        try:
            ckpt_path = os.path.join(config.CKPT_PATH, 'lightning_logs', 'version_0', 'checkpoints')
            if os.path.exists(ckpt_path):
                ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith('.ckpt')]
                if ckpt_files:
                    model = PIClassifier.load_from_checkpoint(os.path.join(ckpt_path, ckpt_files[0]))
                else:
                    raise FileNotFoundError(f'No checkpoint files found in {ckpt_path}')
            else:
                raise FileNotFoundError(f'Checkpoint directory not found: {ckpt_path}')
        except Exception as e:
            print(f'Error loading checkpoint: {e}')
            print('Using freshly initialized model instead')
            model = PIClassifier()

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f'\nLoading STR data from: {config.DEV_LABEL_CSV}')
    dev_df = pd.read_csv(config.DEV_LABEL_CSV)

    pair_ids = dev_df['PairID'].tolist()
    texts1, texts2 = [], []
    for idx, row in dev_df.iterrows():
        text1, text2 = row['Text'].split('\n')
        texts1.append(text1)
        texts2.append(text2)

    print(f'Total pairs: {len(pair_ids)}')

    print('\nComputing PI scores (bidirectional averaging)...')
    pi_scores = []

    with torch.no_grad():
        for x1, x2 in zip(texts1, texts2):
            xs1 = tokenizer.batch_encode_plus(
                [(x1, x2)],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH
            )
            xs2 = tokenizer.batch_encode_plus(
                [(x2, x1)],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH
            )

            xs1 = {k: v.to(device) for k, v in xs1.items()}
            xs2 = {k: v.to(device) for k, v in xs2.items()}

            y1_ = model.model(**xs1, labels=None).logits.softmax(dim=1)[0][1].item()
            y2_ = model.model(**xs2, labels=None).logits.softmax(dim=1)[0][1].item()

            s = (y1_ + y2_) / 2
            pi_scores.append(s)

    results_df = pd.DataFrame({
        'PairID': pair_ids,
        'Pred_Score': pi_scores
    })

    results_df.to_csv(config.RESULTS_CSV, index=False)
    print(f'\nPredictions saved to: {config.RESULTS_CSV}')

    helper.zip_file(config.RESULTS_CSV, config.RESULTS_ZIP)
    print(f'Zip file created: {config.RESULTS_ZIP}')

    return results_df


if __name__ == '__main__':
    config = Config()
    helper.init_logger(config)

    print('Paraphrase Identification (PI) Method for STR Task')
    print('=' * 50)
    for k, v in config.__dict__.items():
        print(f'{k}: {v}')
    print('=' * 50)

    model = train_pi_model(config)

    if model is not None:
        infer_pi_model(config, model)
