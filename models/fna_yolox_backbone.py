import torch

from mmdet.models.builder import BACKBONES

from .fna_base_backbone import BaseBackbone

@BACKBONES.register_module
class YOLOXBackbone(BaseBackbone):
    def __init__(self, search_params, output_indices=[2, 3, 5, 7], init_cfg=None):
        super(YOLOXBackbone, self).__init__(search_params, output_indices, init_cfg)

    def forward(self, inputs):
        net_sub_obj = torch.tensor(0., dtype=torch.float).cuda()

        outs = []
        results = self.blocks[0](inputs)
        results = self.blocks[1](results)

        for i, block in enumerate(self.blocks[2:]):
            results, block_sub_obj = block(results, self.alphas_normal[i], 
                                    self.alphas_reduce[i], self.alphas_index[i], 
                                    self.sub_obj_list[i])
            net_sub_obj += block_sub_obj
            
            if i+2 in self.output_indices:
                outs.append(results)

        if len(outs) == 1:
            return outs[0], net_sub_obj
        else:
            assert len(outs) == 4
            return tuple(outs), net_sub_obj
