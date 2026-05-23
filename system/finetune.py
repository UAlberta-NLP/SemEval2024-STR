#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

"""
Unified Fine-tuning Script for Semantic Textual Relatedness (STR)

Consolidates fine-tuning logic for multiple model backbones (MPNet, T5, GPT-2, RoBERTa)
into a single parameterized script. Differences between models are encoded in
MODEL_CONFIGS dict, keeping training logic model-agnostic.

Supported models:
  - mpnet: sentence-transformers/all-mpnet-base-v2 (contrastive loss)
  - t5: t5-base (regression head with MSE loss)
  - gpt2: gpt2 (regression head with MSE loss)
  - roberta: roberta-base (regression head with MSE loss)

Usage:
  python finetune.py --model_name mpnet --track a --tgt_lan eng --seed 0
  python finetune.py --model_name t5 --track a --tgt_lan eng --seed 0
  python finetune.py --model_name gpt2 --track a --tgt_lan eng --seed 0
  python finetune.py --model_name roberta --track a --tgt_lan eng --seed 0

Model-specific hyperparameters (batch_size, epochs, max_length) are applied
automatically based on --model_name selection.
"""

# dependency
import os
import random
import argparse
import csv
# public
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import evaluation, losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
# private
from config import Config
from src.utils import eva, helper


# Model-specific hyperparameters
MODEL_CONFIGS = {
    'mpnet': {
        'model_id': 'sentence-transformers/all-mpnet-base-v2',
        'method_name': 'sbert',
        'batch_size': 32,
        'epochs': 32,
        'max_length': 156,
        'learning_rate': 5e-5,
        'special_setup': None,
        'framework': 'sentence-transformers'
    },
    't5': {
        'model_id': 't5-base',
        'method_name': 't5',
        'batch_size': 20,
        'epochs': 16,
        'max_length': 256,
        'learning_rate': 2e-5,
        'special_setup': None,
        'framework': 'huggingface'
    },
    'gpt2': {
        'model_id': 'gpt2',
        'method_name': 'gpt2',
        'batch_size': 24,
        'epochs': 24,
        'max_length': 256,
        'learning_rate': 2e-5,
        'special_setup': lambda tokenizer: setattr(tokenizer, 'pad_token', tokenizer.eos_token),
        'framework': 'huggingface'
    },
    'roberta': {
        'model_id': 'roberta-base',
        'method_name': 'roberta',
        'batch_size': 24,
        'epochs': 24,
        'max_length': 256,
        'learning_rate': 2e-5,
        'special_setup': None,
        'framework': 'huggingface'
    }
}

MIX = False


def parse_args():
    """Parse command-line arguments, overlaying on top of config.py"""
    parser = argparse.ArgumentParser(description='Fine-tune models for STR task')
    parser.add_argument('--model_name', type=str, default='mpnet',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model name or HuggingFace model ID')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override default batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override default epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override default learning rate')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Override default max sequence length')
    # Config args (passed through)
    parser.add_argument('--track', type=str, default='a')
    parser.add_argument('--tgt_lan', type=str, default='eng')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def finetune_sentence_transformers(config, model_name, model_id, batch_size, epochs, max_length, lr):
    """Fine-tune sentence-transformers model (MPNet) with contrastive loss"""
    print('\n' + '='*50)
    print(f'Fine-tuning {model_name} with Sentence-Transformers')
    print('='*50)

    # Load model
    model = SentenceTransformer(model_id)

    # Data loading
    eng_train_csv = './res/data/a/eng/eng_train.csv'
    eng_dev_label_csv = './res/data/a/eng/eng_dev_with_labels.csv'

    eng_train_df = pd.read_csv(eng_train_csv)
    eng_train_xs1, eng_train_xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in eng_train_df.iterrows()]))
    eng_train_ys = eng_train_df['Score'].tolist()

    eng_dev_df = pd.read_csv(eng_dev_label_csv)
    eng_dev_xs1, eng_dev_xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in eng_dev_df.iterrows()]))
    eng_dev_ys = eng_dev_df['Score'].tolist()

    # Non-English data augmentation with translations
    if config.tgt_lan != 'eng':
        raw_train_df = pd.read_csv(config.TRAIN_CSV)
        raw_train_ys = raw_train_df['Score'].tolist()
        raw_train_trans_df = pd.read_csv(os.path.join(config.DATA_PATH, 'trans', f'{config.tgt_lan}2eng_train.csv'))
        raw_train_xs1, raw_train_xs2 = map(list, zip(*[(row['Text1 Translation'], row['Text2 Translation']) for idx, row in raw_train_trans_df.iterrows()]))

        raw_dev_df = pd.read_csv(config.DEV_LABEL_CSV)
        raw_dev_ys = raw_dev_df['Score'].tolist()
        raw_dev_trans_df = pd.read_csv(os.path.join(config.DATA_PATH, 'trans', f'{config.tgt_lan}2eng_dev.csv'))
        raw_dev_xs1, raw_dev_xs2 = map(list, zip(*[(row['Text1 Translation'], row['Text2 Translation']) for idx, row in raw_dev_trans_df.iterrows()]))

        dev_size = len(raw_dev_ys)
        if MIX:
            raw_train_xs1 += raw_dev_xs1 + eng_train_xs1 + eng_dev_xs1
            raw_train_xs2 += raw_dev_xs2 + eng_train_xs2 + eng_dev_xs2
            raw_train_ys += raw_dev_ys + eng_train_ys + eng_dev_ys
        else:
            raw_train_xs1 += eng_train_xs1 + eng_dev_xs1
            raw_train_xs2 += eng_train_xs2 + eng_dev_xs2
            raw_train_ys += eng_train_ys + eng_dev_ys
    else:
        dev_size = len(eng_dev_ys)
        if MIX:
            raw_train_xs1 = eng_train_xs1 + eng_dev_xs1
            raw_train_xs2 = eng_train_xs2 + eng_dev_xs2
            raw_train_ys = eng_train_ys + eng_dev_ys
        else:
            raw_train_xs1, raw_train_xs2 = eng_train_xs1, eng_train_xs2
            raw_dev_xs1, raw_dev_xs2 = eng_dev_xs1, eng_dev_xs2
            raw_train_ys, raw_dev_ys = eng_train_ys, eng_dev_ys

    # Train/val split
    if MIX:
        data_size = len(raw_train_ys)
        train_size = data_size - dev_size
        idxes = np.arange(data_size).tolist()
        random.shuffle(idxes)
        train_idxes = idxes[:train_size]
        train_xs1, train_xs2, train_ys = map(list, zip(*[(x1, x2, y) for i, (x1, x2, y) in enumerate(zip(raw_train_xs1, raw_train_xs2, raw_train_ys)) if i in train_idxes]))
        val_xs1, val_xs2, val_ys = map(list, zip(*[(x1, x2, y) for i, (x1, x2, y) in enumerate(zip(raw_train_xs1, raw_train_xs2, raw_train_ys)) if i not in train_idxes]))
    else:
        train_xs1, train_xs2, train_ys = raw_train_xs1, raw_train_xs2, raw_train_ys
        val_xs1, val_xs2, val_ys = raw_dev_xs1, raw_dev_xs2, raw_dev_ys

    print(f'Target Language: {config.tgt_lan}')
    print(f'MIX: {MIX}')
    print(f'Train Size: {len(train_xs1)}')
    print(f'Val Size: {len(val_xs1)}')

    # Build datasets using sentence-transformers v3 API
    from datasets import Dataset as HFDataset
    train_hf = HFDataset.from_dict({'sentence1': train_xs1, 'sentence2': train_xs2, 'label': train_ys})
    val_hf = HFDataset.from_dict({'sentence1': val_xs1, 'sentence2': val_xs2, 'label': val_ys})

    # Loss and evaluator
    train_loss = losses.CosineSimilarityLoss(model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_xs1, val_xs2, val_ys)

    # Fine-tune
    print(f'\nFine-tuning with: batch_size={batch_size}, epochs={epochs}, lr={lr}')
    training_args = SentenceTransformerTrainingArguments(
        output_dir=config.CKPT_PATH,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        adam_epsilon=1e-8,
        warmup_steps=128,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='spearman_cosine',
        report_to='none',
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()
    model.save(config.CKPT_PATH)
    print(f'\nTraining completed! Model saved to: {config.CKPT_PATH}')


def finetune_huggingface(config, model_name, model_id, batch_size, epochs, max_length, lr, special_setup):
    """Fine-tune HuggingFace model (T5, GPT-2) with regression head and MSE loss"""
    print('\n' + '='*50)
    print(f'Fine-tuning {model_name} with HuggingFace Trainer')
    print('='*50)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if special_setup:
        special_setup(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    if hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load data
    labels = []
    sentences = []
    dev_size = 0

    with open(config.TRAIN_CSV, encoding='utf-8', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for count, item in enumerate(reader, 1):
            if count > 1:
                sentence_a = item[1].split('\n')[0]
                sentence_b = item[1].split('\n')[1]
                label = float(item[2])
                labels.append(label)
                sentences.append(sentence_a + '      ' + sentence_b)

    with open(config.DEV_LABEL_CSV, encoding='utf-8', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for count, item in enumerate(reader, 1):
            if count > 1:
                dev_size += 1
                sentence_a = item[1].split('\n')[0]
                sentence_b = item[1].split('\n')[1]
                label = float(item[2])
                labels.append(label)
                sentences.append(sentence_a + '      ' + sentence_b)

    # Train/val split
    data_size = len(sentences)
    train_size = data_size - dev_size
    idxes = np.arange(data_size).tolist()
    random.shuffle(idxes)

    train_idxes = idxes[:train_size]
    train_sentences, train_labels = map(list, zip(*[(sentences[i], labels[i]) for i in idxes if i in train_idxes]))
    val_sentences, val_labels = map(list, zip(*[(sentences[i], labels[i]) for i in idxes if i not in train_idxes]))

    print(f'Target Language: {config.tgt_lan}')
    print(f'Train Size: {len(train_sentences)}')
    print(f'Val Size: {len(val_sentences)}')

    # Tokenize
    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_sentences, truncation=True, padding=True, max_length=max_length)

    # Dataset class
    class STRDataset(Dataset):
        """PyTorch Dataset for STR regression task."""
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = STRDataset(train_encodings, train_labels)
    val_dataset = STRDataset(val_encodings, val_labels)

    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        return {'spearman': eva.get_spearman_cor(labels, predictions)}

    # Training arguments
    args = TrainingArguments(
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        report_to='none',
        optim='adamw_torch',
        save_total_limit=1,
        weight_decay=0.001,
        output_dir=config.CKPT_PATH,
        load_best_model_at_end=True,
        metric_for_best_model='spearman'
    )

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f'\nFine-tuning with: batch_size={batch_size}, epochs={epochs}, max_length={max_length}, lr={lr}')
    trainer.train()
    print(f'\nTraining completed! Model saved to: {config.CKPT_PATH}')


if __name__ == '__main__':
    # Parse args
    args = parse_args()

    # Load config
    config = Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Initialize logger
    helper.init_logger(config)

    # Get model config
    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f'Unknown model: {args.model_name}. Choose from {list(MODEL_CONFIGS.keys())}')

    model_config = MODEL_CONFIGS[args.model_name]
    model_id = model_config['model_id']
    framework = model_config['framework']
    batch_size = args.batch_size or model_config['batch_size']
    epochs = args.epochs or model_config['epochs']
    max_length = args.max_length or model_config['max_length']
    learning_rate = args.learning_rate or model_config['learning_rate']
    special_setup = model_config.get('special_setup')

    # Update config with the actual model and method so CKPT_PATH is correct
    config.update_config(model=model_id, method=model_config['method_name'])

    print('='*50)
    print(f'Fine-tuning Configuration')
    print('='*50)
    print(f'Model: {args.model_name} ({model_id})')
    print(f'Framework: {framework}')
    print(f'Batch Size: {batch_size}')
    print(f'Epochs: {epochs}')
    print(f'Max Length: {max_length}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Track: {config.track}')
    print(f'Language: {config.tgt_lan}')
    print(f'Seed: {config.seed}')
    print(f'Checkpoint Path: {config.CKPT_PATH}')
    print('='*50)

    # Fine-tune
    if framework == 'sentence-transformers':
        finetune_sentence_transformers(config, args.model_name, model_id, batch_size, epochs, max_length, learning_rate)
    elif framework == 'huggingface':
        finetune_huggingface(config, args.model_name, model_id, batch_size, epochs, max_length, learning_rate, special_setup)
    else:
        raise ValueError(f'Unknown framework: {framework}')
