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

def get_model(config):
    import base
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if config.method == 'base':
        return base.Model()
    elif config.method in ['t5', 'gpt2', 'roberta']:
        # Load fine-tuned regression models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(config.CKPT_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(config.CKPT_PATH).to(device)
        model.eval()

        # Wrapper class for fine-tuned models
        class FinetunedModel:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device

            def predict(self, xs1, xs2):
                predictions = []
                with torch.no_grad():
                    for x1, x2 in zip(xs1, xs2):
                        text = f"{x1}      {x2}"
                        encodings = self.tokenizer(
                            text, truncation=True, padding=True, max_length=256, return_tensors='pt'
                        )
                        encodings = {k: v.to(self.device) for k, v in encodings.items()}
                        outputs = self.model(**encodings)
                        score = outputs.logits.squeeze().item()
                        # Clip score to [0, 1] range
                        score = max(0.0, min(1.0, score))
                        predictions.append(score)
                return predictions

        return FinetunedModel(model, tokenizer, device)
    else:
        raise NotImplementedError(f"Method '{config.method}' not implemented in get_model(). Use method-specific scripts: pi.py, nli.py, finetune.py")

def seed_everything(seed: int):
    """
    Seed everything to ensure reproducibility.
    
    Parameters:
    seed (int): The seed value to use for seeding.
    """
    random.seed(seed)
    np.random.seed(seed)