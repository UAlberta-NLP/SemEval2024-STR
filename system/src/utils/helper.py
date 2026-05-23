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

METHOD_MODEL_IDS = {
    'sbert': 'sentence-transformers/all-mpnet-base-v2',
    't5': 't5-base',
    'gpt2': 'gpt2',
    'roberta': 'roberta-base',
}

def get_model(config):
    import base
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if config.method == 'base':
        return base.Model()

    elif config.method == 'sbert':
        from sentence_transformers import SentenceTransformer
        model_id = METHOD_MODEL_IDS['sbert']
        model_name = model_id.replace('/', '_')
        ckpt_path = os.path.join(
            config.RESOURCE_PATH, 'ckpts', config.track, config.tgt_lan,
            'sbert', model_name, str(config.seed)
        )
        model = SentenceTransformer(ckpt_path)
        model.eval()

        class SBERTModel:
            def __init__(self, model):
                self.model = model

            def predict(self, xs1, xs2):
                embeddings1 = self.model.encode(xs1, convert_to_tensor=True)
                embeddings2 = self.model.encode(xs2, convert_to_tensor=True)
                from sentence_transformers import util
                scores = util.cos_sim(embeddings1, embeddings2).diagonal().tolist()
                return [max(0.0, min(1.0, s)) for s in scores]

        return SBERTModel(model)

    elif config.method in ['t5', 'gpt2', 'roberta']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_id = METHOD_MODEL_IDS[config.method]
        model_name = model_id.replace('/', '_')
        ckpt_path = os.path.join(
            config.RESOURCE_PATH, 'ckpts', config.track, config.tgt_lan,
            config.method, model_name, str(config.seed)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
        model.eval()

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