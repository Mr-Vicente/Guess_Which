
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class Guesser_Bot(nn.Module):
    def __init__(self):
        pass

    def train(self, data):
        pass

class Image_Guesser(nn.Module):
    def __init__(self, in_dim, hid_dim, n_images, dropout):
        super(Image_Guesser, self).__init__()
        self.dense_input = nn.Linear(in_dim, hid_dim)
        self.activation_input = nn.ReLU()
        self.norm_input = nn.BatchNorm1d(100)
        self.drop_input = nn.Dropout(dropout, inplace=True)
        self.dense_out = nn.Linear(hid_dim, n_images)
        self.probs = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dense_input(x)
        x = self.activation_input(x)
        x = self.norm_input(x)
        x = self.drop_input(x)
        x = self.dense_out(x)
        prediction = self.probs()(x)
        return prediction