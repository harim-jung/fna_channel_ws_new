import torch.nn as nn

from tools.utils import parse_net_config, sort_net_config
from .operations import OPS, conv_bn, Conv1_1


class Block(nn.Module):
    def __init__(self, in_ch, block_ch, ops, stride, use_se=False, bn_params=[True, True]):
        super(Block, self).__init__()
        layers = []

        for i, op in enumerate(ops):
            if i == 0:
                block_stride = stride
                block_in_ch = in_ch
            else:
                block_stride = 1
                block_in_ch = block_ch
            block_out_ch = block_ch

            layers.append(OPS[op](block_in_ch, block_out_ch, block_stride, 1, 
                                        affine=bn_params[0], track=bn_params[1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def derive_blocks(net_config, if_sort=True):
    """
    net_config=[[in_ch, out_ch], [ops], stride]
    [[32, 16], ['k3_e1'], 1]|
    [[16, 24], ['k5_e6', 'skip', 'skip', 'k3_e6'], 2]|
    [[24, 32], ['k5_e6', 'skip', 'skip', 'k3_e6'], 2]|
    [[32, 64], ['k7_e6', 'k5_e6', 'k3_e6', 'k3_e6'], 2]|
    [[64, 96], ['k3_e6', 'skip', 'skip', 'k7_e6'], 1]|
    [[96, 160], ['k7_e6', 'k7_e6', 'k7_e6', 'k7_e6'], 2]|
    [[160, 320], ['k7_e6'], 1]
    """
    parsed_net_config = parse_net_config(net_config)
    if if_sort:
        parsed_net_config = sort_net_config(parsed_net_config)
    blocks = nn.ModuleList()
    blocks.append(conv_bn(3, parsed_net_config[0][0][0], 2))  # conv_bn(inp, oup, stride)
    for cfg in parsed_net_config:
        if cfg[1][0] == 'conv_2d_1x1':
            blocks.append(Conv1_1(cfg[0][0], cfg[0][1]))
        else:
            # Block(in_ch, block_ch, ops, stride, use_se=False, bn_params=[True, True])
            blocks.append(Block(cfg[0][0], cfg[0][1], cfg[1], cfg[2]))
    return blocks, parsed_net_config
