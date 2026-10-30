import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1
        )


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class ImprovedLocationPredictor(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.5):
        super(ImprovedLocationPredictor, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        self.objectness_head = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.xy_head = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        objectness = self.objectness_head(x)
        xy = self.xy_head(x)
        pred_loc = torch.cat((objectness, xy), dim=1)

        return pred_loc


class ChannelAttention(nn.Module):

    def __init__(self, feat_dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // reduction, feat_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.size()
        y = self.avg_pool(x.permute(0, 2, 1))  # [B, F, 1]
        y = y.view(B, F)  # [B, F]
        y = self.fc(y)  # [B, F]
        y = y.view(B, 1, F)  # [B, 1, F]
        return x * y.expand(-1, T, -1)  # [B, T, F]
