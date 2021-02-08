import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import argparse
import pandas as pd
import spacy
from models import *
from dataset import Seq2SeqDataset, SRC, TRG, load_csv
from datasets import load_metric
from transformers import *
from helper import *
import dill
nlp = spacy.load("en_core_web_sm")

def tokenize_en(text):
	return [tok.text for tok in nlp.tokenizer(text)]

if __name__ == "__main__":

	# arg = argparse.ArgumentParser()
	# arg.add_argument('--description', help='Description Text', default="Search all Wilmington NC homes for sale by map, community", type=str, required=True)
	# args = arg.parse_args()
	
	print('Enter the description...')
	sentence = input()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sentence = tokenize_en(sentence)
	enc_drop = 0.5
	dec_drop = 0.5

	train_descs, train_slogans, valid_descs, valid_slogans, test_descs, test_slogans = load_csv('datasets')

	train_data = Seq2SeqDataset(train_descs, train_slogans, (SRC, TRG))
	test_data = Seq2SeqDataset(test_descs, test_slogans, (SRC, TRG))
	valid_data = Seq2SeqDataset(valid_descs, valid_slogans, (SRC, TRG))

	with open("outputs/SRC.Field", "rb")as f:
		SRC = dill.load(f)

	with open("outputs/TRG.Field", "rb")as f:
		TRG = dill.load(f)

	# SRC.build_vocab(train_data, max_size=30000)
	# TRG.build_vocab(train_data, max_size=30000)

	src_vocab = len(SRC.vocab)
	trg_vocab = len(TRG.vocab)

	TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
	SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
	criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

	attn = Attention(512, 512)
	enc = Encoder(src_vocab, 200, 512, 512, enc_drop)
	dec = Decoder(trg_vocab, 200, 512, 512, dec_drop, attn, True)

	model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
	model.load_state_dict(torch.load('outputs/seq2seq.pt'))

	translation, attention = translate_sentence(sentence, SRC, TRG, model, device)
	display_attention(sentence, translation, attention)