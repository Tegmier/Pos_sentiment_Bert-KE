import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class FinetuneBertFANAttention(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim, embedding_dim):
        super(FinetuneBertFANAttention, self).__init__()
        self.bert = bert_model
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.Feedforward_1 = nn.Linear(embedding_dim, embedding_dim)
        self.Feedforward_2 = nn.Linear(embedding_dim, embedding_dim)
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
        # self.w1 = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, inputs):
        # input_ids: batchsize * sentence_length
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        # embeddings = (batchsize, sequence_len, 768)
        embeddings = outputs.last_hidden_state

        # out = (batch, seq, y/z_dim)
        Feedforward_layer_1 = self.Feedforward_1(embeddings)
        Feedforward_layer_2 = self.Feedforward_2(self.relu(Feedforward_layer_1))
        Feedforward_output = self.layer_norm(embeddings + Feedforward_layer_2)
        #加入注意力计算模块得到注意力

        out1 = self.classifier_y(Feedforward_output)
        out2 = self.classifier_z(Feedforward_output)
        return out1, out2, attention_cal_recursive(Feedforward_output)

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def cal_cos_sim(x1, x2, dim=-1, eps=1e-8):
    dot_product = torch.sum(x1 * x2, dim=dim)
    
    # 计算范数
    norm_x1 = torch.norm(x1, dim=dim)
    norm_x2 = torch.norm(x2, dim=dim)
    
    # 计算余弦相似度
    cos_sim = dot_product / (norm_x1 * norm_x2 + eps)
    
    return cos_sim

def calcos(tensor1, tensor2, eps=1e-8):
    dot_product = torch.sum(tensor1*tensor2)
    
    # 计算范数
    norm_x1 = torch.norm(tensor1)
    norm_x2 = torch.norm(tensor2)
    
    # 计算余弦相似度
    cos_sim = dot_product / (norm_x1 * norm_x2 + eps)
    
    return cos_sim

def attention_cal(data):
    seq_len = data.size(1)
    # X_unsqueeze_1 形状为 (batch, seq, 1, embedding)
    data_unsqueeze_1 = data.unsqueeze(2)
    # X_unsqueeze_2 形状为 (batch, 1, seq, embedding)
    data_unsqueeze_2 = data.unsqueeze(1)
    # 得到(batch, seq, seq)
    cos_sim = cal_cos_sim(data_unsqueeze_1, data_unsqueeze_2, dim=-1)
    attention_weight = torch.sum(cos_sim, dim=-1)/seq_len
    attention_weight = F.softmax(anti_sigmoid(attention_weight), dim=-1)
    return attention_weight

def attention_cal_recursive(data):
    batch_size = data.size(0)
    seq_len = data.size(1)
    cos_sim = torch.zeros(batch_size,seq_len, seq_len).cuda()
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(seq_len):
                tensor1 = data[i,j,:]
                tensor2 = data[i,k,:]
                cos_sim_value = calcos(tensor1,tensor2)
                cos_sim[i,j,k] = cos_sim_value   
    # 得到(batch, seq, seq)
    attention_weight = torch.sum(cos_sim, dim=-1)/seq_len
    attention_weight = F.softmax(anti_sigmoid(attention_weight), dim=-1)
    return attention_weight

def anti_sigmoid(x):
    return 1/(1+torch.exp(x))

