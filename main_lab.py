import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import time
import torch.optim as optim
import numpy as np
from transformers import DistilBertModel
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig


import warnings

# 禁用特定类型的警告，例如 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type='nf4',  # 使用 nf4 量化类型
#     bnb_4bit_use_double_quant=True,  # 启用双量化
#     bnb_4bit_compute_dtype=torch.bfloat16  # 使用 bfloat16 进行计算
# )

# bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased',
#                                             # quantization_config=bnb_config,
#                                             device_map='cuda:0')
bert_model = BertModel.from_pretrained('bert-base-uncased',
                                    #    quantization_config=bnb_config,
                                       device_map='cuda:0')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Global Training Parameters
embedding_dim = bert_model.config.hidden_size

# print(embedding_dim)
y_dim = 2
z_dim = 5
batchsize = 20
nepochs = 50
labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
lr = 0.00001
max_grad_norm = 5
numberofdata = 20000
world_size = 4  # 使用的 GPU 数量


with open('tokenized.pkl', 'rb') as file:
    tokenzied_data = pickle.load(file)

tokenized_tweet_list, tokenized_tag_list = tokenzied_data

with open('qualified_data.pkl', 'rb') as file:
    qualified_data = pickle.load(file)

qualified_tweet_list, qualified_tag_list = qualified_data


data = []
for i in range(len(qualified_tweet_list)):
    data.append([qualified_tweet_list[i], qualified_tag_list[i]])

# 实现通过Dataloader读取数据的方法
from torch.utils.data import Dataset
class KE_Dataloader(Dataset):
    def __init__(self, data) -> None:
        self.tweets = [data[i][0] for i in range(len(data))]
        self.tags = [data[i][1] for i in range(len(data))]

    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, index):
        return self.tweets[index], self.tags[index]
        
def batch_padding_tokenizing_collate_function(batch):
    tweets, tags = zip(*batch)
    tweets = list(tweets)
    tags = list(tags)
    encoded_tweet = tokenizer.batch_encode_plus(
            tweets,
            is_split_into_words=False,  # 指示输入已经是分好词的
            add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
            padding=True,  # 填充到最大长度
            truncation=True,  # 截断超过最大长度的句子
            return_tensors='pt'  # 返回PyTorch张量
        )
    tag_tensor = []
    for t in tags:
        encoded_tag = tokenizer.encode_plus(
            t,
            is_split_into_words=False,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors='pt'
            )
        tag_tensor.append(encoded_tag["input_ids"][0])
    
    y, z = get_label(encoded_tweet, tag_tensor)
    inputs = {}
    inputs["input_ids"] = encoded_tweet["input_ids"]
    inputs["attention_mask"] = encoded_tweet["attention_mask"]
    inputs["label_y"] = torch.tensor(y, dtype=torch.int64)
    inputs["label_z"] = torch.tensor(z, dtype=torch.int64)
    return inputs
    
def iterData(data, batchsize):
    bucket = random.sample(data, len(data))
    bucket = [bucket[i:i+batchsize] for i in range(0, len(bucket), batchsize)]
    random.shuffle(bucket)
    for batch in bucket:
        yield data_tokenizing(batch)

def data_tokenizing(data):
    batch_size = 0
    tweet, tag = [], []
    for tweet_tag in data:
        batch_size+=1
        tweet.append(tweet_tag[0])
        tag.append(tweet_tag[1])
    encoded_tweet = tokenizer.batch_encode_plus(
        tweet,
        is_split_into_words=False,  # 指示输入已经是分好词的
        add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
        padding=True,  # 填充到最大长度
        truncation=True,  # 截断超过最大长度的句子
        return_tensors='pt'  # 返回PyTorch张量
    )
    tag_tensor = []
    for t in tag:
        encoded_tag = tokenizer.encode_plus(
        t,
        is_split_into_words=False,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors='pt'
    )
        tag_tensor.append(encoded_tag["input_ids"][0])
    y, z = get_label(encoded_tweet, tag_tensor)
    inputs = {}
    inputs["input_ids"] = encoded_tweet["input_ids"]
    inputs["attention_mask"] = encoded_tweet["attention_mask"]
    inputs["label_y"] = torch.tensor(y, dtype=torch.int64)
    inputs["label_z"] = torch.tensor(z, dtype=torch.int64)
    return inputs, batch_size

def get_label(encoded_tweet, encoded_tag):
    y , z = [], []
    begin = -1
    # 获取关键词的开始位置
    encoded_tweet_input = encoded_tweet["input_ids"]
    encoded_tag_input = encoded_tag
    for k in range(encoded_tweet_input.size(0)):
        begin = -1
        for i in range(len(encoded_tweet_input[k])):
            the_begining_of_keyphrase_flag = True
            # 从0开始，头到尾把tag长度遍历一遍，如果只是碰巧和前边单词部分重复或者完全不一致,则置flag为false
            for j in range(len(encoded_tag_input[k])):
                if encoded_tweet_input[k][i+j] != encoded_tag_input[k][j]:
                    the_begining_of_keyphrase_flag = False
                    break
            if the_begining_of_keyphrase_flag:
                begin = i
                break
        # 实装label_y
        labels_y = [0]*encoded_tweet_input.size(1)
        for i in range(len(encoded_tag_input[k])):
            labels_y[begin+i] = 1
        y.append(labels_y)
        # 实装label_z 
        # labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
        labels_z = [0]*len(encoded_tweet_input[k])
        if len(encoded_tag_input[k]) == 1:
            labels_z[begin] = labels2idx['S']
        elif len(encoded_tag_input[k]) > 1:
            labels_z[begin] = labels2idx['B']
            for i in range(len(encoded_tag_input[k]) - 2):
                labels_z[begin+i+1] = labels2idx['I']
            labels_z[begin+len(encoded_tag_input[k])-1] = labels2idx['E']
        z.append(labels_z)
    return y, z

class FinetuneBert(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim):
        super(FinetuneBert, self).__init__()
        self.bert = bert_model
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
    
    def forward(self, inputs):
        # input_ids: batchsize * sentence_length
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        # embeddings = (batchsize, sequence_len, 768)
        embeddings = outputs.last_hidden_state

        # 新损失函数
        out1 = self.classifier_y(embeddings)
        out2 = self.classifier_z(embeddings)
        return out1, out2
    
def train_model(model, criterion, optimizer, data_loader, batch_size):
    for epoch in range(nepochs):
        print("Start epoch " + str(epoch) + ":")
        model.train()
        t_start = time.time()
        train_loss = []
        for inputs in data_loader:
            y = inputs["label_y"].cuda()
            z = inputs["label_z"].cuda()
            optimizer.zero_grad()
            y_pred, z_pred = model(inputs)
            y_pred = y_pred.reshape(batch_size, 2, -1)
            z_pred = z_pred.reshape(batch_size, 5, -1)
            loss = (0.5 * criterion(y_pred, y) + 0.5 * criterion(z_pred, z)) / y_pred.size(-1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
            optimizer.step()
            train_loss.append([float(loss), y.size(-1)])
        train_loss = np.array(train_loss)
        train_loss = np.sum(train_loss[:, 0] * train_loss[:, 1]) / np.sum(train_loss[:, 1])
        print('train loss: {:.8f}, time consuming: {}'.format(train_loss, time.time() - t_start))

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# # cuda分布式计算
# # 初始化分布式环境
# def setup(rank, world_size):
#     dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)

# def cleanup():
#     dist.destroy_process_group()

# # 使用DistributedSampler
# def get_data_loader(data, batch_size, rank, world_size):
#     sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
#     data_loader = DataLoader(data, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
#     return data_loader

# 实例化dataloader
from torch.utils.data import DataLoader
dataset = KE_Dataloader(data[:numberofdata])
data_loader = DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    collate_fn=batch_padding_tokenizing_collate_function
)

model = FinetuneBert(bert_model = bert_model, y_dim = y_dim, z_dim = z_dim).cuda()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.AdamW(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr)
model = train_model(model=model, criterion=criterion, optimizer=optimizer, data_loader=data_loader, batch_size=batchsize)