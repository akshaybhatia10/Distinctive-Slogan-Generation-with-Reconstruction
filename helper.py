import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_metric
from transformers import *
import matplotlib.pyplot as plt
import spacy
import matplotlib.ticker as ticker
from datasets import load_metric
from transformers import *

metric = load_metric('rouge')
nlp = spacy.load("en_core_web_sm")

def init_weights(m):
	for name, param in m.named_parameters():
		if 'weight' in name:
			nn.init.normal_(param.data, mean=0, std=0.01)
		else:
			nn.init.constant_(param.data, 0)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, iterator, optimizer, criterion, clip, teacher_force):
	model.train()
	epoch_loss = 0
	
	for i, batch in enumerate(iterator): 
		src, src_len = batch.src
		src_len = list(src_len.cpu().numpy())
		trg = batch.trg
		
		optimizer.zero_grad()
		
		output = model(src, src_len, trg, teacher_force)        
		output_dim = output.shape[-1]
		
		output = output[1:].view(-1, output_dim)
		trg = trg[1:].view(-1)
		
		loss = criterion(output, trg)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()
		epoch_loss += loss.item()
		
	return epoch_loss / len(iterator)

def evaluate_model(model, iterator, criterion, teacher_force):
	model.eval()
	epoch_loss = 0
	
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			src, src_len = batch.src
			src_len = list(src_len.cpu().numpy())
			trg = batch.trg

			output = model(src, src_len, trg, teacher_force)
			output_dim = output.shape[-1]
			
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)
			loss = criterion(output, trg)
			epoch_loss += loss.item()
		
	return epoch_loss / len(iterator)

def train_model_recons(model, recons, iterator, opt_1, opt_2, criterion, cosine, clip, teacher_force):
	model.train()
	recons.train()
	epoch_loss, lamb = 0, 0.4
	
	for i, batch in enumerate(iterator): 
		src, src_len = batch.src
		src_len = list(src_len.cpu().numpy())
		trg = batch.trg
		
		opt_1.zero_grad()
		opt_2.zero_grad()
		
		output = model(src, src_len, trg, teacher_force) 
		d_slo, d_desc = recons(src, trg, output, model)
		output_dim = output.shape[-1]
		
		output = output[1:].view(-1, output_dim)
		trg = trg[1:].view(-1)
		
		seq2seq_loss = criterion(output, trg)
		recons_loss = (1 - cosine(d_slo, d_desc)).mean()
		loss = seq2seq_loss + lamb * recons_loss
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		torch.nn.utils.clip_grad_norm_(recons.parameters(), clip)
		opt_1.step()
		opt_2.step()
		epoch_loss += loss.item()
		
	return epoch_loss / len(iterator)

def evaluate_model_recons(model, recons, iterator, criterion, cosine, teacher_force):
	model.eval()
	recons.eval()
	epoch_loss, lamb = 0, 0.4
	
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			src, src_len = batch.src
			src_len = list(src_len.cpu().numpy())
			trg = batch.trg

			output = model(src, src_len, trg, teacher_force)
			d_slo, d_desc = recons(src, trg, output, model)
			output_dim = output.shape[-1]
			
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)
			seq2seq_loss = criterion(output, trg)
			recons_loss = (1 - cosine(d_slo, d_desc)).mean()
			loss = seq2seq_loss + lamb * recons_loss
			epoch_loss += loss.item()
		
	return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):

	model.eval()
	if isinstance(sentence, str):
		tokens = [token.text.lower() for token in nlp(sentence)]
	else:
		tokens = [token.lower() for token in sentence]

	tokens = [src_field.init_token] + tokens + [src_field.eos_token]		
	src_indexes = [src_field.vocab.stoi[token] for token in tokens]
	src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
	src_len = torch.LongTensor([len(src_indexes)])
	
	with torch.no_grad():
		encoder_outputs, hidden = model.encoder(src_tensor, src_len)

	mask = model.create_mask(src_tensor) 
	trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
	attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
	
	for i in range(max_len):
		trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)		
		with torch.no_grad():
			output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

		attentions[i] = attention	
		pred_token = output.argmax(1).item()
		trg_indexes.append(pred_token)

		if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
			break
	
	trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
	
	return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def display_attention(sentence, translation, attention):
	
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111)	
	attention = attention.squeeze(1).cpu().detach().numpy()
	cax = ax.matshow(attention, cmap='bone')  
	ax.tick_params(labelsize=15)
	ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], rotation=45)
	ax.set_yticklabels(['']+translation)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	plt.show()
	plt.close()

def calculate_scores(model, SRC, TRG, descriptions, references, device):
	predictions = []
	for desc in descriptions:
		translation, _ = translate_sentence(desc, SRC, TRG, model, device)
		translation = translation[:-1] 
		if '<unk>' in translation:
			translation = list(filter(lambda a: a != '<unk>', translation))
		predictions.append(' '.join(translation))

	scores = metric.compute(predictions=predictions, references=references, rouge_types=['rouge1', 'rouge2', 'rougeL'])
	R1_F1 = scores['rouge1'][1][2]
	R2_F1 = scores['rouge2'][1][2]
	RL_F1 = scores['rougeL'][1][2]

	return R1_F1, R2_F1, RL_F1