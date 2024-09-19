from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoModel, AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 导入原始 BERT 模型
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 导入 RoBERTa 模型
roberta_model = RobertaModel.from_pretrained('roberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 导入 ALBERT 模型
albert_model = AlbertModel.from_pretrained('albert-base-v2')
albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# 导入 DistilBERT 模型
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 导入 TinyBERT 模型
tinybert_model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
tinybert_tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

# 导入 BERTweet 模型
bertweet_model = AutoModel.from_pretrained('Twitter/twhin-bert-base')
bertweet_tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
