import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(input_channels, output_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.relu(x)

class GNNModel(torch.nn.Module):
    def __init__(self, input_channels):
        super(GNNModel, self).__init__()
        self.layer1 = GCNLayer(input_channels, 16)   # 16可以根据你的需求进行更改
        self.layer2 = GCNLayer(16, 1)  # 输出层

    def forward(self, data):
        x = self.layer1(data.x, data.edge_index)
        x = self.layer2(x, data.edge_index)
        return x
