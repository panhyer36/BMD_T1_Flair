import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    Lightweight CNN model for BMD regression prediction
    Input: [batch, 15, 256, 256]  # 15 middle slices as channels
    Output: [batch, 1] BMD prediction value
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature extraction layers (reduced channels)
        self.features = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(15, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling + Regression head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x


class ResNetBMD(nn.Module):
    """
    ResNet18-based model for BMD regression prediction
    Uses pretrained ImageNet weights with modified first conv for 15 channels
    Input: [batch, 15, 256, 256]
    Output: [batch, 1] BMD prediction value
    """

    def __init__(self):
        super(ResNetBMD, self).__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv: 3 channels -> 15 channels
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy 3-channel weights 5 times to fill 15 channels
            self.conv1.weight[:, :3] = old_conv.weight
            self.conv1.weight[:, 3:6] = old_conv.weight
            self.conv1.weight[:, 6:9] = old_conv.weight
            self.conv1.weight[:, 9:12] = old_conv.weight
            self.conv1.weight[:, 12:15] = old_conv.weight

        # Copy remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Replace FC layer with regression head
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Test ResNetBMD model
    print("Testing ResNetBMD:")
    print("=" * 50)
    model = ResNetBMD()

    # Test forward pass
    dummy_input = torch.randn(4, 15, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
