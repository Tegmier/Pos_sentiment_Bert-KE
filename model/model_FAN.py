import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig

class FinetuneBertFAN(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim, embedding_dim):
        super(FinetuneBertFAN, self).__init__()
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
        return out1, out2

bert_model = BertModel.from_pretrained('bert-base-uncased',
                                    #    quantization_config=bnb_config,
                                       device_map='cuda:0')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


