import torch.nn as nn
from transformers import BertModel, BertTokenizer

class FinetuneBertMFANrnn(nn.Module):
    def __init__(self, bert_model, y_dim, z_dim, embedding_dim):
        super(FinetuneBertMFANrnn, self).__init__()
        self.bert = bert_model
        self.relu = nn.ReLU()
        self.num_layer = 12
        self.lstm_num_layer = 2
        self.lstem_dropout = 0.1
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.Feedforward_1 = nn.Linear(embedding_dim, embedding_dim)
        self.Feedforward_2 = nn.Linear(embedding_dim, embedding_dim)
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=self.lstm_num_layer,
            bidirectional=False,
            batch_first=True,
            dropout=self.lstem_dropout
        )
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        # self.w1 = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, inputs):
        # input_ids: batchsize * sentence_length
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        # embeddings = (batchsize, sequence_len, 768)
        embeddings = outputs.last_hidden_state

        # out = (batch, seq, y/z_dim)
        # for _ in range(self.num_layer):
        #     embeddings = self.layer_norm(embeddings + self.Feedforward_2(self.relu(self.Feedforward_1(embeddings))))
        out1, _ = self.rnn(embeddings)
        out1 = self.fc(out1)
        out1 = self.layer_norm(embeddings + self.Feedforward_2(self.relu(self.Feedforward_1(out1))))
        out2, _ = self.rnn(out1)
        out2 = self.fc(out2)
        out2 = self.layer_norm(embeddings + self.Feedforward_2(self.relu(self.Feedforward_1(out2))))
        #加入注意力计算模块得到注意力
        out1 = self.classifier_y(embeddings)
        out2 = self.classifier_z(embeddings)
        return out1, out2

bert_model = BertModel.from_pretrained('bert-base-uncased', hidden_dropout_prob=0.1
                                    #    quantization_config=bnb_config,
                                    )
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


