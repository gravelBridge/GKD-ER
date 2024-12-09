import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNetSmall(nn.Module):
    def __init__(self, output_dim=10):
        super(ResNetSmall, self).__init__()
        self.layer1 = BasicBlock(1, 64)
        self.layer2 = BasicBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.layer3 = BasicBlock(64, 64)
        self.layer4 = BasicBlock(64, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(64*7*7, 256)
        self.classifier = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, return_features=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.pool1(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)
        feat = out.view(out.size(0), -1)
        feat = self.dropout(self.relu(self.fc(feat)))
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        else:
            return logits

class GKD_ER_Full_Model(nn.Module):
    def __init__(self, output_dim=10, num_tasks=5, alpha=0.1):
        super(GKD_ER_Full_Model, self).__init__()
        self.backbone = ResNetSmall(output_dim=output_dim)
        self.hidden_dim = 256
        self.num_tasks = num_tasks
        self.scaling_vectors = nn.Parameter(torch.zeros(num_tasks, self.hidden_dim))
        self.active_task = None
        self.alpha = alpha
    
    def set_active_task(self, t_id):
        self.active_task = t_id
    
    def forward(self, x, return_features=False):
        logits, feat = self.backbone(x, return_features=True)
        if self.active_task is not None:
            s = self.scaling_vectors[self.active_task]
            feat = feat * (1 + self.alpha * s)
            logits = self.backbone.classifier(feat)
        if return_features:
            return logits, feat
        else:
            return logits
