
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 32 Rec.field: 1 JumpOut: 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.x1 =nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 32 Rec.field: 1 JumpOut: 1
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.R1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 16 Rec.field: 1 JumpOut: 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 16 Rec.field: 1 JumpOut: 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

#         self.layer1=nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 16 Rec.field: 1 JumpOut: 1
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),# input: 3x3 output: 16 Rec.field: 1 JumpOut: 1
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.x2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 8 Rec.field: 1 JumpOut: 1
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 4 Rec.field: 1 JumpOut: 1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # input: 3x3 output: 4 Rec.field: 1 JumpOut: 1
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=3, stride=1, padding=1),# input: 3x3 output: 4 Rec.field: 1 JumpOut: 1
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(1, 10)

    def forward(self, x):
        out= self.input_layer(x);
        X = self.x1(out)
        R = self.R1(X)
        out = self.x1(out)+self.R1(X)
        out = self.layer2(out)
        X = self.x2(out)
        R = self.R2(X)
        out = self.x2(out) + self.R2(X)
        out = self.pool1(out) 
        out = self.linear(out)
        return out
        # out = out.view(-1, 10)
#         return F.log_softmax(out, dim=-1)


def ResNet18():
    return ResNet()


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
