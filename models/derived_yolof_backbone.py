import torch.nn as nn

from mmcv.cnn import kaiming_init
from mmdet.models.builder import BACKBONES

from .derive_blocks import derive_blocks
from tools.apis.param_remap import remap_for_paramadapt


@BACKBONES.register_module
class FNA_Yolof(nn.Module):
    def __init__(self, net_config, pretrained=None, output_indices=[8]):
        super(FNA_Yolof, self).__init__()
        self.blocks, _ = derive_blocks(net_config)
        self.pretrained = pretrained
        self.output_indices = output_indices

    def forward(self, x, stat=None):
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.output_indices:
                outs.append(x)
        
        return tuple(outs)

    def train(self, mode=True):
        super(FNA_Yolof, self).train(mode)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.pretrained is not None and self.pretrained.use_load:
            model_dict = remap_for_paramadapt(self.pretrained.load_path, self.state_dict(), self.pretrained.seed_num_layers)
            self.load_state_dict(model_dict)