from utils.model.conv1d_list import *
from utils.model.conv1d_seq import *
import torch
from torch import nn

class Model1(nn.Module):
    def __init__(self, feature_dim, num_layer = 3, conv_hidden_dim=512, out_dim=115, dr_rate=3):
        super(Model1, self).__init__()
        self.pre_conv = Conv1d_seq(feature_dim, conv_hidden_dim)
        self.conv_layer = Conv1d_list(num_layer, conv_hidden_dim, conv_hidden_dim)
        self.fc1 = nn.Linear(conv_hidden_dim, conv_hidden_dim)
        self.fc2 = nn.Linear(conv_hidden_dim, out_dim)
        self.drop_out = nn.Dropout(p=dr_rate * 0.01)
        self.relu = nn.ReLU()

    def forward(self, feature):
        logits = feature.transpose(-1, -2)
        logits = self.pre_conv(logits)
        logits = self.conv_layer(logits)
        logits = self.fc1(torch.mean(logits, dim=-1))
        logits = self.relu(logits)
        logits = self.drop_out(logits)
        logits = self.fc2(logits)
        logits = self.drop_out(logits)
        return logits

class Model2(nn.Module):
    def __init__(self, feature_dim, conv_hidden_dim=512, out_dim=115, dr_rate=30):
        super(Model2, self).__init__()
        self.conv_layer = nn.ModuleList([
            Conv1d_seq_Batch_Avg(in_channel=feature_dim, out_channel=feature_dim, padding="same", padding_mode="circular"),
            Conv1d_seq_Batch_Avg(in_channel=feature_dim, out_channel=conv_hidden_dim, padding="same", padding_mode="circular"),
            Conv1d_seq_Batch_Avg(in_channel=conv_hidden_dim, out_channel=conv_hidden_dim, padding="same", padding_mode="circular"),
            Conv1d_seq_Batch_Avg(in_channel=conv_hidden_dim, out_channel=feature_dim, padding="same", padding_mode="circular"),
        ])
        self.fc1 = nn.Linear(feature_dim, int(conv_hidden_dim * 0.5))
        self.fc2 = nn.Linear(int(conv_hidden_dim * 0.5), out_dim)
        self.drop_out = nn.Dropout(p=dr_rate * 0.01)
        self.relu = nn.ReLU()
    
    def forward(self, feature):
        logits = feature.transpose(-1, -2)
        for layer in self.conv_layer:
            logits = layer(logits)
        logits = self.fc1(torch.mean(logits, dim=-1))
        logits = self.relu(logits)
        logits = self.drop_out(logits)
        logits = self.fc2(logits)
        logits = self.drop_out(logits)
        return logits

class Model3(nn.Module):
    def __init__(self, feature_dim, conv_hidden_dim=512, out_dim=115, dr_rate=30):
        super(Model3, self).__init__()
        self.pre_conv = Conv1d_seq(in_channel = feature_dim,out_channel= conv_hidden_dim, padding="same")
        self.conv_layer = Conv1d_list_Batch_Avg(num_layer=3, in_channel=conv_hidden_dim, out_channel=conv_hidden_dim, padding="same", drop_rate=0.4)
        self.fc1 = nn.Linear(conv_hidden_dim, int(conv_hidden_dim * 0.5))
        self.fc2 = nn.Linear(int(conv_hidden_dim * 0.5), out_dim)
        self.drop_out = nn.Dropout(p=dr_rate * 0.01)
        self.relu = nn.ReLU()

    def forward(self, feature):
        logits = feature.transpose(-1, -2)
        logits = self.pre_conv(logits)
        logits = self.conv_layer(logits)
        logits = self.fc1(torch.mean(logits, dim=-1))
        logits = self.relu(logits)
        logits = self.drop_out(logits)
        logits = self.fc2(logits)
        logits = self.drop_out(logits)
        return logits

class Model4(nn.Module):
    def __init__(self, Nframe, feature_dim, conv_hidden_dim=512, out_dim=115, dr_rate=30):
        self.nframe = Nframe
        self.feature_dim = feature_dim
        super(Model4, self).__init__()
        self.conv_layer = nn.ModuleList([
            Conv1d_seq_Batch_Avg(in_channel=feature_dim, out_channel=feature_dim, padding="same", padding_mode="circular"),
            Conv1d_seq_Batch_Avg(in_channel=feature_dim, out_channel=conv_hidden_dim, padding="same", padding_mode="circular"),
            Conv1d_seq_Batch_Avg(in_channel=conv_hidden_dim, out_channel=conv_hidden_dim, padding="same", padding_mode="circular"),
            Conv1d_seq_Batch_Avg(in_channel=conv_hidden_dim, out_channel=feature_dim, padding="same", padding_mode="circular"),
        ])
        self.fc1 = nn.Linear((Nframe - 8) * feature_dim, conv_hidden_dim * 10)
        self.fc2 = nn.Linear(conv_hidden_dim * 10, out_dim)
        self.drop_out = nn.Dropout(p=dr_rate * 0.01)
        self.relu = nn.ReLU()
    
    def forward(self, feature):
        logits = feature.transpose(-1, -2)
        for layer in self.conv_layer:
            logits = layer(logits)
        # print(logits.shape)
        logits = self.fc1(logits.view(-1, (self.nframe-8) * self.feature_dim))
        logits = self.relu(logits)
        logits = self.drop_out(logits)
        logits = self.fc2(logits)
        logits = self.drop_out(logits)
        return logits