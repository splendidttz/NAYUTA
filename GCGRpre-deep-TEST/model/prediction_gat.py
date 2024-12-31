import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn.conv import MessagePassing

class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(CustomGATConv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.bias = nn.Parameter(torch.Tensor(heads * out_channels)) if bias and concat else nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, edge_index_i, x_i, x_j, num_nodes):
        x_j = torch.cat([x_i, x_j], dim=-1).view(-1, self.heads, 2 * self.out_channels)
        alpha = (x_j * self.att).sum(dim=-1)
        alpha = torch.nn.functional.leaky_relu(alpha, self.negative_slope)
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        alpha = torch.nn.functional.softmax(alpha, dim=1)
        return x_j[:, :, self.out_channels:] * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        if self.concat:
            return aggr_out.view(-1, self.heads * self.out_channels) + self.bias
        else:
            return aggr_out.mean(dim=1) + self.bias

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})'

    def get_attention_weights(self):
        return self.att

class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, output_dim=128, dropout=0.2, heads=4):
        super(GATNet, self).__init__()
        self.conv1 = CustomGATConv(num_features_xd, num_features_xd, heads=heads, concat=True, dropout=dropout)
        self.conv2 = CustomGATConv(num_features_xd * heads, num_features_xd * 2, heads=heads, concat=True, dropout=dropout)
        self.conv3 = CustomGATConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads, concat=True, dropout=dropout)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = gmp(x, batch)  # 使用全局最大池化
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
