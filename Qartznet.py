import torch
from torch import nn


class tcsc(nn.Module):
    def __init__(self, inn, out, kernel_size, stride=1, dilation=1):
        super(tcsc, self).__init__()

        self.tcs_segment = nn.Sequential(

            nn.Conv1d(inn, inn, kernel_size, dilation=dilation, stride=stride,  # depthwise conv
                      groups=inn, padding=dilation * kernel_size // 2),

            nn.Conv1d(inn, out, 1),  # pointwise conv

            nn.BatchNorm1d(out),

            nn.ReLU(True),
        )

    def forward(self, x):
        return self.tcs_segment(x)


class tcsc(nn.Module):
    def __init__(self, inn, out, kernel_size, stride=1, dilation=1):
        super(tcsc, self).__init__()

        self.tcs_segment = nn.Sequential(

            nn.Conv1d(inn, inn, kernel_size, dilation=dilation, stride=stride,  # depthwise conv
                      groups=inn, padding=dilation * kernel_size // 2),

            nn.Conv1d(inn, out, 1),  # pointwise conv

            nn.BatchNorm1d(out),

            nn.ReLU(True),
        )

    def forward(self, x):
        return self.tcs_segment(x)


class tcsc_last(nn.Module):
    def __init__(self, inn, out, kernel_size, stride=1, dilation=1):
        super(tcsc_last, self).__init__()

        self.tcs_segment = nn.Sequential(

            nn.Conv1d(inn, inn, kernel_size, dilation=dilation, stride=stride,  # depthwise conv
                      groups=inn, padding=dilation * kernel_size // 2),

            nn.Conv1d(inn, out, 1),  # pointwise conv

            nn.BatchNorm1d(out),

        )

    def forward(self, x):
        return self.tcs_segment(x)


class Block(nn.Module):

    def __init__(self, inn, out, kernel_size, stride=1, dilation=1):
        super(Block, self).__init__()

        self.pointwise = nn.Conv1d(inn, out, 1)

        self.batch_norm = nn.BatchNorm1d(out)

        self.all_tcs = nn.Sequential(

            tcsc(inn, out, kernel_size, stride=stride, dilation=dilation),

            tcsc(out, out, kernel_size, stride=stride, dilation=dilation),

            tcsc(out, out, kernel_size, stride=stride, dilation=dilation),

            tcsc(out, out, kernel_size, stride=stride, dilation=dilation),

            tcsc_last(out, out, kernel_size, stride=stride, dilation=dilation)

        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.pointwise(x)
        y = self.batch_norm(y)
        x = self.all_tcs(x)
        x = x + y
        x = self.relu(x)

        return x


class QNet(nn.Module):
    def __init__(self, n_mels, n_classes):
        super(QNet, self).__init__()
        self.n_classes = n_classes

        self.net = nn.Sequential(

            nn.Conv1d(n_mels, n_mels, 33, groups=n_mels, stride=2, padding=33 // 2),  # C1
            nn.Conv1d(n_mels, 256, 1),

            Block(256, 256, 33),  # Block1

            Block(256, 256, 39),  # Block2

            Block(256, 512, 51),  # Block3

            Block(512, 512, 63),  # Block4

            Block(512, 512, 75),  # Block5

            nn.Conv1d(512, 512, 87, groups=512, dilation=2, padding=87 - 1),  # C2
            nn.Conv1d(512, 512, 1),

            nn.Conv1d(512, 1024, 1),  # C3

            nn.Conv1d(1024, n_classes, 1),  # C4

            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        return self.net(x)