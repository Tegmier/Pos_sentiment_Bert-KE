import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig
import torch.nn.functional as F

class FinetuneBertRelu(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim, embedding_dim, hidden_dim):
        super(FinetuneBertRelu, self).__init__()
        self.bert = bert_model
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(embedding_dim, hidden_dim)
        self.classifier_y = nn.Linear(hidden_dim, y_dim)
        self.classifier_z = nn.Linear(hidden_dim, z_dim)
        # self.w1 = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, inputs):
        # input_ids: batchsize * sentence_length
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        # embeddings = (batchsize, sequence_len, 768)
        embeddings = outputs.last_hidden_state

        # out = (batch, seq, y/z_dim)
        hidden = self.relu(self.hidden_layer(embeddings))
        out1 = self.classifier_y(hidden)
        out2 = self.classifier_z(hidden)
        return out1, out2

bert_model = BertModel.from_pretrained('bert-base-uncased',
                                    #    quantization_config=bnb_config,
                                       device_map='cuda:0')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')