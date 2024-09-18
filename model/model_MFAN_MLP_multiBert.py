import torch.nn as nn
import torch.nn.functional as F

class MLPFANBlock(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super(MLPFANBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_num_layer = 3
        self.Feedforward_1 = nn.Linear(embedding_dim, embedding_dim)
        self.Feedforward_2 = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.MLPlist = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(self.hidden_num_layer)])
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(embedding_dim)
    
    def forward(self, inputs):
        for layer in self.MLPlist:
            inputs = layer(inputs)
        inputs =  self.layernorm(self.Feedforward_2(self.relu(self.Feedforward_1(inputs)))+inputs)
        return inputs


class MultiBERT(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim, embedding_dim):
        super(MultiBERT, self).__init__()
        self.bert = bert_model
        self.mlp_num_layer = 3
        self.embedding_dim = embedding_dim
        self.blocklist_1 = nn.ModuleList([MLPFANBlock(embedding_dim) for _ in range(self.mlp_num_layer)])
        self.blocklist_2 = nn.ModuleList([MLPFANBlock(embedding_dim) for _ in range(self.mlp_num_layer)])
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
    
    def forward(self, inputs):
        # input_ids: batchsize * sentence_length
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        # embeddings = (batchsize, sequence_len, 768)
        embeddings = outputs.last_hidden_state

        # out = (batch, seq, y/z_dim)
        for mlpblock in self.blocklist_1:
            embeddings = mlpblock(embeddings)
        
        out1 = self.classifier_y(embeddings)

        for mlpblock in self.blocklist_2:
            embeddings = mlpblock(embeddings)
        
        out2 = self.classifier_z(embeddings)
        return out1, out2


