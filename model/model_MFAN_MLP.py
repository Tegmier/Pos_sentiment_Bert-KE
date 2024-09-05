import torch.nn as nn
from transformers import BertModel, BertTokenizer

class FinetuneBertMFANMLP(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim, embedding_dim):
        super(FinetuneBertMFANMLP, self).__init__()
        self.bert = bert_model
        self.relu = nn.ReLU()
        self.MLP_num_layer = 6
        self.FAN_num_layer = 6
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.Feedforward_1 = nn.Linear(embedding_dim, embedding_dim)
        self.Feedforward_2 = nn.Linear(embedding_dim, embedding_dim)
        self.MLP = nn.Linear(embedding_dim, embedding_dim)
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
        for _ in range(self.FAN_num_layer):
            for _ in range(self.MLP_num_layer):
                embeddings = self.MLP(embeddings)
            embeddings = self.layer_norm(embeddings + self.Feedforward_2(self.relu(self.Feedforward_1(embeddings))))
        
        #加入注意力计算模块得到注意力
        out1 = self.classifier_y(embeddings)

        for _ in range(self.FAN_num_layer):
            for _ in range(self.MLP_num_layer):
                embeddings = self.MLP(embeddings)
            embeddings = self.layer_norm(embeddings + self.Feedforward_2(self.relu(self.Feedforward_1(embeddings))))
        
        out2 = self.classifier_z(embeddings)
        return out1, out2

bert_model = BertModel.from_pretrained('bert-base-uncased', hidden_dropout_prob=0.1,
                                    #    quantization_config=bnb_config,
                                       device_map='cuda:0')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


