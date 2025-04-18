import torch
import torch.nn as nn
import torch.nn.functional as F


def init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0)
        nn.init.constant_(layer.bias.data, 0)


class MLP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLP, self).__init__()
        self.ln1 = nn.Linear(in_channels, num_classes)
        init(self.ln1)

    def forward(self, x):
        out = self.ln1(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Mappatura della connessione residuale
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Connessione residuale
        out = self.relu(out)
        return out


# Definizione della ResNet 1D
class ResNet1D(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet1D, self).__init__()

        # Layer iniziali
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Blocchi residuali
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Fully connected
        self.fc = nn.Linear(512, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionTime(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, num_filters):
        super(InceptionTime, self).__init__()
        self.inception_blocks = nn.ModuleList()
        self.in_channels = in_channels

        for _ in range(num_blocks):
            in_channels = self.in_channels
            self.in_channels = num_filters * 4
            self.inception_blocks.append(
                InceptionModule(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
            )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        for inception_block in self.inception_blocks:
            x = inception_block(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
