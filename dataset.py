import torchtext
from torchtext import data
from torchtext.data import Field, BucketIterator
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")


def tokenize_en(text):
	return [tok.text for tok in nlp.tokenizer(text)]

def load_csv(path):
	train = pd.read_csv(path + '/train.csv', encoding='utf-8')
	valid = pd.read_csv(path + '/valid.csv', lineterminator='\n')
	test = pd.read_csv(path + '/test.csv', lineterminator='\n')

	train = train.sample(100)
	test = test.sample(100)
	valid = valid.sample(100)

	train_descs = list(train['description'].values)
	train_slogans = list(train['slogan'].values)

	valid_descs = list(valid['description'].values)
	valid_slogans = list(valid['slogan'].values)

	test_descs = list(test['decription'].values)
	test_slogans = list(test['slogan'].values)

	return train_descs, train_slogans, valid_descs, valid_slogans, test_descs, test_slogans

SRC = Field(tokenize = tokenize_en, 
			init_token = '<sos>', 
			eos_token = '<eos>',
			include_lengths = True, 
			lower = False)

TRG = Field(tokenize = tokenize_en, 
			init_token = '<sos>', 
			eos_token = '<eos>', 
			lower = False)

class Seq2SeqDataset(data.Dataset):
	def __init__(self, description, slogan, fields, **kwargs):
		examples = []
		fields = [('src', fields[0]), ('trg', fields[1])]

		for d, s in zip(description, slogan):
			examples.append(data.Example.fromlist([d, s], fields))
		super(Seq2SeqDataset, self).__init__(examples, fields, **kwargs)