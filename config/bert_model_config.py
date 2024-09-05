# Config Bert model
from transformers import BertTokenizer, BertModel

BERT_MODEL_NAME = 'bert-base-uncased'

bertmodel = BertModel.from_pretrained(BERT_MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)