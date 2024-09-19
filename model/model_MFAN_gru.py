import torch.nn as nn
from transformers import BertModel, BertTokenizer    

class MLPFANBlock(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super(MLPFANBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_num_layer = 1
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

class FinetuneBertMFANgru(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim, embedding_dim):
        super(FinetuneBertMFANgru, self).__init__()
        self.embedding_dim = embedding_dim
        self.bert = bert_model
        self.lstm_num_layer = 2
        self.lstem_dropout = 0.1
        # self.mlpblock1 = MLPFANBlock(self.embedding_dim)
        # self.mlpblock2 = MLPFANBlock(self.embedding_dim)
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
        self.bilstm1 = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=self.lstm_num_layer,
            bidirectional=False,
            batch_first=True,
            dropout=self.lstem_dropout
        )
        self.bilstm2 = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=self.lstm_num_layer,
            bidirectional=False,
            batch_first=True,
            dropout=self.lstem_dropout
        )
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, inputs):
        # input_ids: batchsize * sentence_length
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        embeddings = outputs.last_hidden_state

        out1, _ = self.bilstm1(embeddings)
        out1 = self.fc1(out1)
        # out1 = self.mlpblock1(out1)

        out2, _ = self.bilstm2(out1)
        out2 = self.fc2(out2)
        # out2 = self.mlpblock2(out2)

        out1 = self.classifier_y(out1)
        out2 = self.classifier_z(out2)
        return out1, out2

bert_model = BertModel.from_pretrained('bert-base-uncased', hidden_dropout_prob=0.1
                                    #    quantization_config=bnb_config,
                                    )
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


