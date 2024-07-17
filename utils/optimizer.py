"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel
from sentence_transformers import SentenceTransformer
from transformers import RobertaConfig

BERT_CLASS = {
    "distilbert": 'models/distilbert-base-uncased',
    "vascl":'models/asw-ai-vascl-base',
    "vascll": 'models/asw-ai-vascl',
}

SBERT_CLASS = {
    "distilbert": 'models/distilbert-base-nli-stsb-mean-tokens',
}

VASCL_CLASS = {
    "vascl": 'models/asw-ai-vascl-base',
    "vascll": 'models/asw-ai-vascl',
}

def get_optimizer(model, args):
    
    optimizer = torch.optim.Adam([
        {'params':model.bert.parameters()}, 
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}
    ], lr=args.lr)
    
    print(optimizer)
    return optimizer 
    

def get_bert(args):
    
    if args.use_pretrain == "SBERT":
        bert_model = get_sbert(args)
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")

    elif args.use_pretrain == "VASCL":
        tokenizer = AutoTokenizer.from_pretrained(VASCL_CLASS[args.bert])
        config = RobertaConfig.from_pretrained(VASCL_CLASS[args.bert])
        model = RobertaModel.from_pretrained(VASCL_CLASS[args.bert], config=config)
        if args.bert == "vascl":
            print("..... loading VASCL-Roberta-base !!!")
        else:
            print("..... loading VASCL-Roberta-large !!!")
    else:
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        print("..... loading plain BERT !!!")
        
    return model, tokenizer


def get_sbert(args):
    sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    return sbert








