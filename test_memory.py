import torch
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import time
import torch.optim as optim
import numpy as np
from transformers import DistilBertModel
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type='nf4',  # 使用 nf4 量化类型
#     bnb_4bit_use_double_quant=True,  # 启用双量化
#     bnb_4bit_compute_dtype=torch.bfloat16  # 使用 bfloat16 进行计算
# )

bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                            # quantization_config=bnb_config,
                                            device_map='cuda:0')

tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

sentence = "I like to eat some fish"
encoded_sentence = tokenizer.encode_plus(
    sentence,
    is_split_into_words=False,  # 指示输入已经是分好词的
    add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
    padding=False,  # 填充到最大长度
    truncation=False,  # 截断超过最大长度的句子
    return_tensors='pt'  # 返回PyTorch张量
)
inputs = {}
inputs["input_ids"] = encoded_sentence["input_ids"]
inputs["attention_mask"] = encoded_sentence["attention_mask"]
print(inputs)

class FinetuneTest(nn.Module):
    def __init__(self, bert_model):
        super(FinetuneTest, self).__init__()
        self.bert = bert_model
    
    def forward(self, inputs):
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        return outputs

model = FinetuneTest(bert_model=bert_model).cuda()
model.train()
embeddings = model(inputs)
print(embeddings)
