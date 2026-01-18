import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )
        # 2. define full-connected layer to classify
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10)   # CIFAR-10 -> 10 classes
        )
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        # classification
        out = self.classifier(x)
        return out


class ResBlock(nn.Module):
    ''' residual block'''
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        # 1. define double convolution
        # convolution
        # batch normalization
        # activate function
        # ......
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                    stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                    stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if in_channel != out_channel or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)
        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        out = self.conv_layers(x)
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        # 3. Add the output of the convolution and the original data (or from 2.)
        out = out + identity
        # 4. relu
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    '''residual network'''
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,
                    padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 2. define multiple residual blocks
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))        
        # 3. define full-connected layer to classify
        self.fc = nn.Linear(256, 10)

        # ===== [MODIFICATION] =====
        # Store last convolutional feature maps for CAM
        self.feature_maps = None
        # ==========================

        # ===== [ADDED FOR GRAD-CAM] =====
        self.gradients = None
        # =================================


    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # ===== [MODIFICATION] =====
        # Save feature maps before global average pooling
        self.feature_maps = x
        # ==========================

        # ===== [ADDED FOR GRAD-CAM] =====
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        # =================================

        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        # classification
        out = self.fc(x)

        return out
    
    # ===== [ADDED FOR GRAD-CAM] =====
    def save_gradient(self, grad):
        self.gradients = grad
    # =================================

    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.
        hidden_channel = out_channel // bottle_neck

        # 1. define convolution
        # 1x1 convolution
        # batch normalization
        # activate function
        # 3x3 convolution
        # ......
        # 1x1 convolution
        # ......
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3,
                    stride=stride, padding=1, groups=group, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                        stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = None

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        out = self.conv(x)
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        # 3. Add the output of the convolution and the original data (or from 2.)
        out = out + identity
        # 4. relu
        out = self.relu(out)
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,
                    padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        bottle_neck = 4
        group = 32
        # 2. define multiple residual blocks
        self.layer1 = nn.Sequential(
            ResNextBlock(64, 128, bottle_neck, group, stride=1),
            ResNextBlock(128, 128, bottle_neck, group, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResNextBlock(128, 256, bottle_neck, group, stride=2),
            ResNextBlock(256, 256, bottle_neck, group, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResNextBlock(256, 512, bottle_neck, group, stride=2),
            ResNextBlock(512, 512, bottle_neck, group, stride=1),
        )
        # 3. define full-connected layer to classify
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)

        x = torch.flatten(x, 1)      

        # classification
        out = self.fc(x)  
        return out

