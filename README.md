### Official implementation of:

**Harim Jung, Myeong-Seok Oh, Cheoljong Yang, Seong-Whan Lee, “Neural Architecture Adaptation for Object Detection by Searching Channel Dimensions and Mapping Pre-trained Parameters,” ICPR 2022.** [[Paper](https://arxiv.org/pdf/2206.08509)]

This project introduces a channel-adaptive neural architecture search (NAS) framework for object detection. The key innovation lies in searching for optimal channel dimensions, operations and the number of layers, and **mapping pre-trained parameters** to the newly searched architecture, enabling efficient adaptation without training from scratch.

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
