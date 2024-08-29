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
        return out1, out2, attention_cal(Feedforward_output)

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def attention_cal(data):
    seq_len = data.size(1)
    # X_unsqueeze_1 形状为 (batch, seq, 1, embedding)
    data_unsqueeze_1 = data.unsqueeze(2)
    # X_unsqueeze_2 形状为 (batch, 1, seq, embedding)
    data_unsqueeze_2 = data.unsqueeze(1)
    # 得到(batch, seq, seq)
    cos_sim = F.cosine_similarity(data_unsqueeze_1, data_unsqueeze_2, dim=-1)
    attention_weight = torch.sum(cos_sim, dim=-1)/seq_len
    attention_weight = F.softmax(anti_sigmoid(attention_weight), dim=-1)
    return attention_weight

def anti_sigmoid(x):
    return 1/(1+torch.exp(x))

