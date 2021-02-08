import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
		super().__init__()
		self.embedding = nn.Embedding(input_dim, emb_dim)
		self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
		self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, src, src_len):
		embedded = self.dropout(self.embedding(src))                
		packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)    
		packed_outputs, hidden = self.rnn(packed_embedded)
		outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
		hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
		
		return outputs, hidden

class Attention(nn.Module):
	def __init__(self, enc_hid_dim, dec_hid_dim):
		super().__init__()
		self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
		self.v = nn.Linear(dec_hid_dim, 1, bias = False)
		
	def forward(self, hidden, encoder_outputs, mask):        
		batch_size = encoder_outputs.shape[1]
		src_len = encoder_outputs.shape[0]
		
		hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
		encoder_outputs = encoder_outputs.permute(1, 0, 2)
		energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
		attention = self.v(energy).squeeze(2)       
		attention = attention.masked_fill(mask == 0, -1e10)
		
		return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
	def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, copy=False):
		super().__init__()
		self.output_dim = output_dim
		self.attention = attention
		self.embedding = nn.Embedding(output_dim, emb_dim)
		self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
		self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
		self.p_gen_W = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, 1)
		self.dropout = nn.Dropout(dropout)
		self.copy = copy
		print ('Training model with copy mechanism: ', self.copy)
		
	def forward(self, input, hidden, encoder_outputs, mask):        
		input = input.unsqueeze(0)        
		embedded_ = self.dropout(self.embedding(input))
			
		a_ = self.attention(hidden, encoder_outputs, mask)        
		a = a_.unsqueeze(1)        
		encoder_outputs = encoder_outputs.permute(1, 0, 2)        
		weighted = torch.bmm(a, encoder_outputs)        
		weighted_ = weighted.permute(1, 0, 2)        
		rnn_input = torch.cat((embedded_, weighted_), dim = 2)
			
		output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
		assert (output == hidden).all()
		
		embedded = embedded_.squeeze(0)
		output = output.squeeze(0)
		weighted = weighted_.squeeze(0)
		
		prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
		
		if self.copy:
			self.p_gen = torch.sigmoid(self.p_gen_W(torch.cat((weighted, hidden.squeeze(0), embedded), dim = 1)))
			self.gen_preds = torch.mul(self.p_gen, prediction) + torch.mul(1 - self.p_gen, torch.sum(a_, dim=1).unsqueeze(1))
			prediction = self.gen_preds
		
		return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, src_pad_idx, device):
		super().__init__()
		
		self.encoder = encoder
		self.decoder = decoder
		self.src_pad_idx = src_pad_idx
		self.device = device
		
	def create_mask(self, src):
		mask = (src != self.src_pad_idx).permute(1, 0)
		return mask
		
	def forward(self, src, src_len, trg, teacher_force):                    
		batch_size = src.shape[1]
		trg_len = trg.shape[0]
		trg_vocab_size = self.decoder.output_dim
		
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
		encoder_outputs, hidden = self.encoder(src, src_len)
				
		input = trg[0,:]
		
		mask = self.create_mask(src)                
		for t in range(1, trg_len):
			output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)            
			outputs[t] = output
			top1 = output.argmax(1) 

			input = trg[t] if teacher_force else top1
			
		return outputs

class Reconstruction(nn.Module):
	def __init__(self, output_dim, embed_dim, dec_hidden, dec_drop, trg_pad_index, device):
		super().__init__()
		self.output_dim = output_dim
		self.embed_dim = embed_dim
		self.dec_hidden = dec_hidden
		self.dec_drop = nn.Dropout(dec_drop)
		self.trg_pad_index = trg_pad_index
		self.device = device

		self.Wa1_slo = nn.Linear(self.dec_hidden, 1)
		self.Wa2_slo = nn.Linear(self.output_dim, self.dec_hidden)
		self.Wc_slo = nn.Linear(self.output_dim, 1)
		self.Wr1_desc = nn.Linear(1, self.dec_hidden)
		self.Wr2_desc = nn.Linear(self.embed_dim, self.dec_hidden)
		self.v = nn.Linear(self.dec_hidden, 1)

	def create_mask(self, src):
		mask = (src != self.trg_pad_index).permute(1, 0)
		return mask

	def forward(self, src, trg, decoder_outputs, model):
		mask = self.create_mask(trg)
		sst_ = F.gumbel_softmax(decoder_outputs)
		sst = sst_.permute(1, 0, 2)
		att_base_ = self.Wa1_slo(torch.tanh(self.Wa2_slo(sst)))
		att_base = att_base_.squeeze(2)
		att_base = att_base.masked_fill(mask == 0, -1e10)
		at_self_ = F.softmax(att_base, dim=1)
		at_self = at_self_.unsqueeze(1)
		rep_slo_ = torch.bmm(at_self, sst)
		rep_slo = rep_slo_.squeeze(1)
		d_slo = self.Wc_slo(rep_slo)

		mask = self.create_mask(src)
		E_ = self.dec_drop(model.encoder.embedding(src))
		E = E_.permute(1, 0, 2)
		u_i_ = torch.tanh(self.Wr1_desc(d_slo) + self.Wr2_desc(E).permute(1, 0, 2))
		u_i = u_i_.permute(1, 0, 2)
		att_vu_ = self.v(u_i)
		att_vu = att_vu_.squeeze(2)
		att_vu = att_vu.masked_fill(mask == 0, -1e10)
		a_refer_ = F.softmax(att_vu, dim=1)
		a_refer = a_refer_.unsqueeze(1)
		d_desc_ = torch.bmm(a_refer, E)
		d_desc = d_desc_.squeeze(1)

		return d_slo, d_desc