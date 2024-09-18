import torch.nn as nn

class MLPFANBlock(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super(MLPFANBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_num_layer = 2
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

class contrast_rnn(nn.Module):
    def __init__(self,vocab_size, y_dim, z_dim, embedding_dim):
        super(contrast_rnn, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_num_layer = 2
        self.rnn_dropout = 0.1
        self.relu = nn.ReLU()
        self.mlp_num_layer = 2
        self.blocklist_1 = nn.ModuleList([MLPFANBlock(embedding_dim) for _ in range(self.mlp_num_layer)])
        self.blocklist_2 = nn.ModuleList([MLPFANBlock(embedding_dim) for _ in range(self.mlp_num_layer)])
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.MLP = nn.Linear(embedding_dim, embedding_dim)
        self.lstm = nn.RNN(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=self.lstm_num_layer,
            bidirectional=False,
            batch_first=True,
            dropout=self.rnn_dropout
        )
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
    
    def forward(self, inputs):
        lex = inputs["lex"].cuda()
        lex = self.embedding(lex)
        lstm_output, _ = self.lstm(lex)
        # y_task
        for mlpblock in self.blocklist_1:
            lstm_output = mlpblock(lstm_output)
        
        out1 = self.classifier_y(lstm_output)

        # z_task
        for mlpblock in self.blocklist_2:
            lstm_output = mlpblock(lstm_output)
        
        out2 = self.classifier_z(lstm_output)
        return out1, out2



