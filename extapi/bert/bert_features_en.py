from typing import List
import sys
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import logging

logging.set_verbosity_error()

# this part of the code creates BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = torch.tensor([1]).unsqueeze(0)


def bert_vector(text:str):
   '''computes vector representation of text using BERT'''
   text = text[0:300]
   inputs = tokenizer(text.lower(), return_tensors="pt")
   outputs = model(**inputs, labels=labels, output_hidden_states=True)
   q = outputs[2][11][0]
   return q.mean(dim=0).cpu().detach().numpy()