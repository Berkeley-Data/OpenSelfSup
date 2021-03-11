import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init

from .registry import INPUT_MODULES
from .utils import build_norm_layer




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@INPUT_MODULES.register_module
class Conv1x1Block(nn.Module):
    """
    Conv1x1 => Batch Norm => RELU input module
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1Block, self).__init__()
        self.net = nn.Sequential(
            conv1x1(in_channels, out_channels), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def init_weights(self, init_linear='normal'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        
                
    def forward(self, x):
        return self.net(x)
