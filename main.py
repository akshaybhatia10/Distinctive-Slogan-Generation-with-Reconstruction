import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import pandas as pd
import spacy
import tqdm
import argparse
import os
import sys
import dill
from dataset import Seq2SeqDataset, SRC, TRG, load_csv
from torchtext.data import Field, BucketIterator
from helper import *
from models import *
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
	arg = argparse.ArgumentParser()
	arg.add_argument('--enc_dec_hidden', help='Number of hidden units for encoder decoder GRUs', default=512, type=int, required=True)
	arg.add_argument('--bs', help='Batch size', default=8, type=int, required=True)
	arg.add_argument('--epochs', help='Number of epochs', default=10, type=int, required=True)
	arg.add_argument('--dataset_path', help='Path to dataset - must have train.csv, test.csv, valid.csv', default='datasets', type=str, required=True)
	arg.add_argument('--vocab_size', help='Size of vocab', default=30000, type=int, required=True)
	arg.add_argument('--embed_size', help='Word Embedding size', default=200, type=int, required=True)
	arg.add_argument('--copy', help='Whether to enable copy mechanism or not', action='store_true')
	arg.add_argument('--recons', help='Whether to enable reconstruction model or not', action='store_true')
	arg.add_argument('--evaluate', help='Evaluate the model using the pretrained model', action='store_true')
	args = arg.parse_args()

	evaluate = True if args.evaluate else False
	copy = True if args.copy else False
	recons = True if args.recons else False

	print ('------------ Loading Datasets ------------\n')
	train_descs, train_slogans, valid_descs, valid_slogans, test_descs, test_slogans = load_csv(args.dataset_path)

	train_data = Seq2SeqDataset(train_descs, train_slogans, (SRC, TRG))
	test_data = Seq2SeqDataset(test_descs, test_slogans, (SRC, TRG))
	valid_data = Seq2SeqDataset(valid_descs, valid_slogans, (SRC, TRG))

	print ('------------ Building Vocab ------------\n')
	
	SRC.build_vocab(train_data, max_size=args.vocab_size)
	TRG.build_vocab(train_data, max_size=args.vocab_size)

	train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
													(train_data, valid_data, test_data), 
													batch_size = args.bs,
													sort_within_batch = True,
													sort_key = lambda x : len(x.src),
													device = device)

	src_vocab = len(SRC.vocab)
	trg_vocab = len(TRG.vocab)
	enc_drop = 0.5
	dec_drop = 0.5

	TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
	SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
	criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

	attn = Attention(args.enc_dec_hidden, args.enc_dec_hidden)
	enc = Encoder(src_vocab, args.embed_size, args.enc_dec_hidden, args.enc_dec_hidden, enc_drop)
	dec = Decoder(trg_vocab, args.embed_size, args.enc_dec_hidden, args.enc_dec_hidden, dec_drop, attn, copy)

	seq2seq = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
	seq2seq.apply(init_weights)
	nll_loss = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
	optimizer_model = optim.Adam(seq2seq.parameters())
	print(f'The seq2seq model has {count_parameters(seq2seq):,} trainable parameters\n')

	if recons:
		crr = Reconstruction(trg_vocab, args.embed_size, args.enc_dec_hidden, dec_drop, TRG_PAD_IDX, device).to(device)
		cosine_loss = torch.nn.CosineSimilarity()
		crr.apply(init_weights)
		optimizer_crr = optim.Adam(crr.parameters())
		print(f'The reconstruction model has {count_parameters(crr):,} trainable parameters')


	if evaluate:
		with open("outputs/SRC.Field", "rb")as f:
			SRC = dill.load(f)

		with open("outputs/TRG.Field", "rb")as f:
			TRG = dill.load(f)

		print ('------------ Evaluating model: outputs/seq2seq.pt ------------\n')
		seq2seq.load_state_dict(torch.load('outputs/seq2seq.pt'))
		R1_F1, R2_F1, RL_F1 = calculate_scores(seq2seq, SRC, TRG, test_descs, test_slogans, device)
		print (f'R1: {R1_F1*100:.4f}, R2: {R2_F1*100:.4f}, RL: {RL_F1*100:.4f}, ')

		sys.exit(0)

	CLIP = 1
	best_valid_loss = float('inf')

	with open("outputs/SRC.Field", "wb")as f:
		dill.dump(SRC, f)

	with open("outputs/TRG.Field", "wb")as f:
		dill.dump(TRG, f)

	print ('------------ Starting Training ------------\n')
	for epoch in range(args.epochs):
		start_time = time.time()
		
		if recons:
			train_loss = train_model_recons(seq2seq, crr, train_iterator, optimizer_model, optimizer_crr, nll_loss, cosine_loss, CLIP, teacher_force=True)
			valid_loss = evaluate_model_recons(seq2seq, crr, valid_iterator, nll_loss, cosine_loss, teacher_force=False)
		else:
			train_loss = train_model(seq2seq, train_iterator, optimizer_model, nll_loss, CLIP, teacher_force=True)
			valid_loss = evaluate_model(seq2seq, valid_iterator, nll_loss, teacher_force=False)
		
		end_time = time.time()
		
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			if recons:
				test_loss = evaluate_model_recons(seq2seq, crr, test_iterator, nll_loss, cosine_loss, teacher_force=False)
				torch.save(crr.state_dict(), 'outputs/crr.pt')
				torch.save(seq2seq.state_dict(), 'outputs/seq2seq.pt')
			else:
				test_loss = evaluate_model(seq2seq, test_iterator, nll_loss, teacher_force=False)
				torch.save(seq2seq.state_dict(), 'outputs/seq2seq.pt')
			print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
		
		print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
