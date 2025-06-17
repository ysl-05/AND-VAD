import torch
import torch.nn as nn
import torch.nn.functional as F


class Featurizer(nn.Module):
    def __init__(self, n_frames):
        super(Featurizer, self).__init__()
        self.conv0 = nn.Conv3d(1, 64, (5, 3, 3), padding=(2, 1, 1))
        self.conv1 = nn.Conv3d(64, 128, (5, 3, 3), padding=(2, 1, 1))
        self.conv2 = nn.Conv3d(128, 128, (5, 3, 3), padding=(2, 1, 1))
        self.conv3 = nn.Conv3d(128, 1, (1, 1, 1))

        self.bn0 = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(1)

        self.pool0 = nn.MaxPool2d((2, 2))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        self.n_outputs = n_frames * 128

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        x = self.pool0(x)

        x = x.reshape(-1, 64, x.size(2), x.size(3))
        x = x.permute(0, 1, 2, 3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        x = self.pool1(x)

        x = x.reshape(-1, 128, x.size(2), x.size(3))
        x = x.permute(0, 1, 2, 3)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# class MLP(nn.Module):
#     def __init__(self, n_outputs):
#         super(MLP, self).__init__()
#         self.inputs = nn.Linear(n_outputs, n_outputs)
#         self.hidden = nn.Linear(n_outputs, n_outputs)
#         self.outputs = nn.Linear(n_outputs, n_outputs)

#     def forward(self, x):
#         x = self.inputs(x)
#         x = F.relu(x)
#         x = self.outputs(x)
#         return x


def Classifier(in_features, out_features, nonlinear=False):
    if nonlinear:
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, out_features)
        )
    else:
        return nn.Linear(in_features, out_features)
