### Official implementation of:

**Harim Jung, Myeong-Seok Oh, Cheoljong Yang, Seong-Whan Lee, “Neural Architecture Adaptation for Object Detection by Searching Channel Dimensions and Mapping Pre-trained Parameters,” ICPR 2022.** [[Paper (arXiv)](https://arxiv.org/pdf/2206.08509)]

Most object detection frameworks use backbone architectures originally designed for image classification, conventionally with pre-trained parameters on ImageNet. However, image classification and object detection are essentially different tasks and there is no guarantee that the optimal backbone for classification is also optimal for object detection. Recent neural architecture search (NAS) research has demonstrated that automatically designing a backbone specifically for object detection helps improve the overall accuracy. In this paper, we introduce **a neural architecture adaptation method that can optimize the given backbone for detection purposes, while still allowing the use of pre-trained parameters**. We propose to adapt both the micro- and macro-architecture by searching for **specific operations and the number of layers**, in addition to the **output channel dimensions** of each block. It is important to find the optimal channel depth, as it greatly affects the feature representation capability and computation cost. We conduct experiments with our searched backbone for object detection and demonstrate that our backbone outperforms both manually designed and searched state-of-the-art backbones on the COCO dataset.

## Install
```
# Create virtual env
conda create -n fna_env python=3.8 -y
conda activate fna_env

# Install PyTorch (change according to your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

# Install MMCV
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

## Run code
```
# search
python tools/search.py --config configs/fna_retinanet_fpn_search.py --data_path /path/to/coco/ --job_name retinanet_search

# train
python tools/train.py --config configs/fna_retinanet_fpn_retrain.py --data_path /path/to/coco/ --job_name retinanet_train

# test
python tools/test.py \
    configs/fna_retinanet_fpn_retrain.py \
    </path/to/checkpoint> \
    --eval bbox \
    --launcher pytorch \
    --work-dir <path/to/output> \
    --out <path/to/output/file>
```
