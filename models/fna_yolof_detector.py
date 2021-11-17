from mmdet.core import bbox2result
from mmdet.models.detectors.yolof import YOLOF
from mmdet.models.builder import DETECTORS
import torch.distributed as dist

@DETECTORS.register_module
# inherits Yolof Class
class NASYolof(YOLOF):
    
    def extract_feat(self, img):
        x, sub_obj = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x, sub_obj

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x, sub_obj = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, 
                                              gt_labels, gt_bboxes_ignore=gt_bboxes_ignore)
        
        return losses, sub_obj

    # def simple_test(self, img, img_meta, rescale=False, **kwargs):
    #     x, _ = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #     bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results[0]
    
    def simple_test(self, img, img_metas, rescale=False):
        feat, _ = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
