import torch
from torch import nn
from utils.model.conv1d_seq import Conv1d_seq_Batch_Avg, Conv1d_seq

class Conv1d_list(nn.Module):
    def __init__(self, num_layer, in_channel, out_channel, kernel_size = 3, stride = 1, padding = 0, padding_mode = "zeros", drop_rate = 0.3):
        super().__init__()
        self.conv_layer = torch.nn.ModuleList([
            Conv1d_seq(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, drop_rate = drop_rate) for _ in range(num_layer)
        ])
    
    def forward(self, x):
        for layer in self.conv_layer:
            x = layer(x)
        return x

class Conv1d_list_Batch_Avg(nn.Module):
    def __init__(self, num_layer, in_channel, out_channel, kernel_size = 3, stride = 1, padding = 0, padding_mode = "zeros", drop_rate = 0.3, AvgPool_kernel_size = 3, AvgPool_stride = 1):
        super().__init__()
        self.conv_layer = torch.nn.ModuleList([
            Conv1d_seq_Batch_Avg(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, drop_rate = drop_rate, AvgPool_kernel_size = AvgPool_kernel_size, AvgPool_stride = AvgPool_stride) for _ in range(num_layer)
        ])
    
    def forward(self, x):
        for layer in self.conv_layer:
            x = layer(x)
        return x
    
