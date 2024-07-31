import torch
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

embedding_dim = 768
y_dim = 2
z_dim = 5
batchsize = 5


with open('tokenized.pkl', 'rb') as file:
    tokenzied_data = pickle.load(file)

tokenized_tweet_list, tokenized_tag_list = tokenzied_data

with open('qualified_data.pkl', 'rb') as file:
    qualified_data = pickle.load(file)

qualified_tweet_list, qualified_tag_list = qualified_data

data = []

# for i in range(len(tokenized_tweet_list)):
#     data.append([tokenized_tweet_list[i], tokenized_tag_list[i]])

for i in range(len(qualified_tweet_list)):
    data.append([qualified_tweet_list[i], qualified_tag_list[i]])

def iterData(data, batchsize):
    bucket = random.sample(data, len(data))
    bucket = [bucket[i:i+batchsize] for i in range(0, len(bucket), batchsize)]
    random.shuffle(bucket)
    for batch in bucket:
        yield data_tokenizing(data)

def data_tokenizing(data):
    tweet, tag = [], []
    for tweet_tag in data:
        tweet.append(tweet_tag[0])
        tag.append(tweet_tag[1])
    encoded_inputs = tokenizer.batch_encode_plus(
        tweet,
        is_split_into_words=False,  # 指示输入已经是分好词的
        add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
        # max_length=10,  # 设置最大长度
        padding=True,  # 填充到最大长度
        truncation=True,  # 截断超过最大长度的句子
        return_tensors='pt'  # 返回PyTorch张量
    )
    print(encoded_inputs['input_ids'])

data_tokenizing(data[0:5])




# class FinetuneBert(nn.moudle):
#     def __int__(self, bert_model, y_dim, z_dim):
#         super(FinetuneBert, self).__init__()
#         self.bert = bert_model
#         self.classifier_y = nn.Linear(embedding_dim, y_dim)
#         self.classifier_z = nn.Linear(embedding_dim, z_dim)
    
#     def forward(self, inputs):
#         input_ids = inputs['input_ids']

#         outputs = self.bert()