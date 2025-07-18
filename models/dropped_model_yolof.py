import torch
import torch.nn as nn
import torch.distributed as dist


class Yolof_Dropped_Network(nn.Module):
    def __init__(self, super_model):
        super(Yolof_Dropped_Network, self).__init__()
        # static modules loading
        self.sub_obj_list = super_model.sub_obj_list
        self.alphas_normal = super_model.alphas_normal
        self.alphas_reduce = super_model.alphas_reduce
        self.alpha_index = super_model.alpha_index
        self.blocks = super_model.blocks
        self.output_indices = super_model.output_indices

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
                                        self.alphas_reduce[i], self.alpha_index[i], 
                                        self.sub_obj_list[i])
                net_sub_obj += block_sub_obj
            
            if i+2 in self.output_indices:
                outs.append(results)
        
        return tuple(outs), net_sub_obj