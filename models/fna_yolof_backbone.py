import torch
import torch.distributed as dist
from mmdet.models.builder import BACKBONES

from .fna_base_backbone import BaseBackbone
from .operations import Conv1_1

@BACKBONES.register_module
class YolofBackbone(BaseBackbone):
    def __init__(self, search_params, output_indices=[8], pretrained=None):
        super(YolofBackbone, self).__init__(search_params, output_indices=output_indices, pretrained=pretrained)
        # add an extra layer for further channel depth expansion to pass as input to dilated encoder of YOLOF
        self.blocks.append(Conv1_1(self.net_scale.chs[-2], self.net_scale.chs[-1]))

    def forward(self, inputs):
        net_sub_obj = torch.tensor(0., dtype=torch.float).cuda()

        results = self.blocks[0](inputs)
        results = self.blocks[1](results)

        outs = []
        for i, block in enumerate(self.blocks[2:]):
            if block == self.blocks[-1]:
                results = self.blocks[-1](results)
            else:
                results, block_sub_obj = block(results, self.alphas_normal[i], 
                                        self.alphas_reduce[i], self.alphas_index[i], 
                                        self.sub_obj_list[i])
                net_sub_obj += block_sub_obj
            
            if i+2 in self.output_indices:
                outs.append(results)

        # results = results.reshape(results.shape[0], 1, results.shape[1],  results.shape[2],  results.shape[3])

        return tuple(outs), net_sub_obj