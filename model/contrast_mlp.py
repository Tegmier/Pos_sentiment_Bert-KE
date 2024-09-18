import torch.nn as nn

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

class contrast_mlp(nn.Module):
    def __init__(self,vocab_size, y_dim, z_dim, embedding_dim):
        super(contrast_mlp, self).__init__()
        self.embedding_dim = embedding_dim
        self.relu = nn.ReLU()
        main_mlp_layer_num = 3
        self.mlp_num_layer = 3
        self.mlp_blocklist = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range (main_mlp_layer_num)])
        self.blocklist_1 = nn.ModuleList([MLPFANBlock(embedding_dim) for _ in range(self.mlp_num_layer)])
        self.blocklist_2 = nn.ModuleList([MLPFANBlock(embedding_dim) for _ in range(self.mlp_num_layer)])
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.classifier_y = nn.Linear(embedding_dim, y_dim)
        self.classifier_z = nn.Linear(embedding_dim, z_dim)
    
    def forward(self, inputs):
        lex = inputs["lex"].cuda()
        lex = self.embedding(lex)        

        for layer in self.mlp_blocklist:
            lex = layer(lex)

        # y_task
        for mlpblock in self.blocklist_1:
            lex = mlpblock(lex)
        
        out1 = self.classifier_y(lex)

        # z_task
        for mlpblock in self.blocklist_2:
            lex = mlpblock(lex)
        
        out2 = self.classifier_z(lex)
        return out1, out2



