import torch.nn as nn


class contrast_lstm(nn.Module):
    def __init__(self,vocab_size, y_dim, z_dim, embedding_dim):
        super(contrast_lstm, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_num_layer = 2
        self.lstm_dropout = 0.1
        self.relu = nn.ReLU()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.FAN_num_layer = 6
        self.MLP_num_layer = 6
        self.MLP = nn.Linear(embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=self.lstm_num_layer,
            bidirectional=False,
            batch_first=True,
            dropout=self.lstm_dropout
        )
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.Feedforward_1 = nn.Linear(embedding_dim, embedding_dim)
        self.Feedforward_2 = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
    
    def forward(self, inputs):
        lex = inputs["lex"].cuda()
        lex = self.embedding(lex)
        lstm_output, _ = self.lstm(lex)
        # 处理embedding的层结束了
        embeddings = self.layer_norm(lstm_output + self.Feedforward_2(self.relu(self.Feedforward_1(lstm_output))))

        # y_task
        for _ in range(self.FAN_num_layer):
            for _ in range(self.MLP_num_layer):
                embeddings = self.MLP(embeddings)
            embeddings = self.layer_norm(embeddings + self.Feedforward_2(self.relu(self.Feedforward_1(embeddings))))
        out1 = self.classifier_y(embeddings)

        # z_task
        for _ in range(self.FAN_num_layer):
            for _ in range(self.MLP_num_layer):
                embeddings = self.MLP(embeddings)
            embeddings = self.layer_norm(embeddings + self.Feedforward_2(self.relu(self.Feedforward_1(embeddings))))
        out2 = self.classifier_z(embeddings)

        return out1, out2



