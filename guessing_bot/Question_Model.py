
import torch.nn as nn

class Questioner(nn.Module):
    def __init__(self, config, token_size):
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=config.WORD_EMBED_SIZE
        )

        self.lstm = nn.LSTM(
            input_size=config.WORD_EMBED_SIZE,
            hidden_size=config.LSTM_OUT_SIZE,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        pass
