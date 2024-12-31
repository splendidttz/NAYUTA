import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class GCNNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, output_dim=256, dropout=0.5):
        super(GCNNet, self).__init__()
        # GCN 层
        self.conv1 = GCNConv(num_features_xd, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, 64)
        self.conv4 = GCNConv(64, 32)

        # 全连接层
        self.fc_g1 = torch.nn.Linear(32, 1024)
        self.fc_g2 = torch.nn.Linear(1024, 512)
        self.fc_g3 = torch.nn.Linear(512, 256)
        self.out = torch.nn.Linear(256, 1)

        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 批归一化层
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)
        self.batch_norm_fc1 = nn.BatchNorm1d(1024)
        self.batch_norm_fc2 = nn.BatchNorm1d(512)
        self.batch_norm_fc3 = nn.BatchNorm1d(256)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN 层
        x = self.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = self.relu(self.batch_norm2(self.conv2(x, edge_index)))
        x = self.relu(self.batch_norm3(self.conv3(x, edge_index)))
        x = self.relu(self.batch_norm4(self.conv4(x, edge_index)))

        # 全局最大池化
        x = gmp(x, batch)

        # 全连接层
        x = self.relu(self.batch_norm_fc1(self.fc_g1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm_fc2(self.fc_g2(x)))
        x = self.dropout(x)
        last_layer_features = self.relu(self.batch_norm_fc3(self.fc_g3(x)))
        x = self.dropout(last_layer_features)

        # 输出层
        x = self.out(x)
        x = torch.sigmoid(x)

        # 返回最终输出和最后一层特征
        return x, last_layer_features
