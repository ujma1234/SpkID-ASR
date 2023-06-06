import torch
from torch import nn

class Conv1d_seq(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1, padding = 0, padding_mode = "zeros", drop_rate = 0.3):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding = padding, padding_mode=padding_mode),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Conv1d_seq_Batch_Avg(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1, padding = 0, padding_mode = "zeros", drop_rate = 0.3, AvgPool_kernel_size = 3, AvgPool_stride = 1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding = padding, padding_mode=padding_mode),
            torch.nn.BatchNorm1d(out_channel),
            torch.nn.Dropout(p=drop_rate),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(AvgPool_kernel_size, AvgPool_stride)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

