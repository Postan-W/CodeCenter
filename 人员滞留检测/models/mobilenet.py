import torch
import torch.nn as nn

class MobileNetV2IFN(nn.Module):
    """
    MobileNetV2 described by Sandler et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
    """

    def __init__(self) -> None:
        super(MobileNetV2IFN, self).__init__()
        self.loss = {'xent'}

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)

        self.block2 = Bottleneck(32, 16, 1, 1, instance_normalization=True)
        self.block3 = nn.Sequential(
            Bottleneck(16, 24, 6, 2, instance_normalization=True),
            Bottleneck(24, 24, 6, 1, instance_normalization=True),
        )
        self.block4 = nn.Sequential(
            Bottleneck(24, 32, 6, 2, instance_normalization=True),
            Bottleneck(32, 32, 6, 1, instance_normalization=True),
            Bottleneck(32, 32, 6, 1, instance_normalization=True),
        )
        self.block5 = nn.Sequential(
            Bottleneck(32, 64, 6, 2, instance_normalization=True),
            Bottleneck(64, 64, 6, 1, instance_normalization=True),
            Bottleneck(64, 64, 6, 1, instance_normalization=True),
            Bottleneck(64, 64, 6, 1, instance_normalization=True),
        )
        self.block6 = nn.Sequential(
            Bottleneck(64, 96, 6, 1, instance_normalization=True),
            Bottleneck(96, 96, 6, 1, instance_normalization=True),
            Bottleneck(96, 96, 6, 1, instance_normalization=True),
        )
        self.block7 = nn.Sequential(
            Bottleneck(96, 160, 6, 2),
            Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 160, 6, 1),
        )
        self.block8 = Bottleneck(160, 320, 6, 1)
        self.conv9 = ConvBlock(320, 1280, 1)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = nn.functional.relu6(self.in1(self.conv1(x)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.conv9(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def load_weights(self, model_path: str) -> None:
        checkpoint = torch.load(model_path)

        # The pretrained model from https://github.com/BJTUJia/person_reID_DualNorm contains unused weights
        # which cause error in load_state_dict(), we delete them to avoid the error.
        for k in ["bnneck.weight", "bnneck.bias", "bnneck.running_mean", "bnneck.running_var",
                  "bnneck.num_batches_tracked", "classifier.weight"]:
            del checkpoint[k]
        self.load_state_dict(checkpoint)
        
    


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution (bias discarded) + batch normalization + relu6.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
        g (int): number of blocked connections from input channels
                 to output channels (default: 1).
    """

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False, groups=g)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return nn.functional.relu6(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, instance_normalization=False):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.IN = None

        if instance_normalization:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwconv2(m)
        m = self.conv3(m)

        if self.use_residual:
            out = x + m
        else:
            out = m

        if self.IN is not None:
            return self.IN(out)
        else:
            return out
