from __future__ import division

import argparse
import os
import os.path as osp
import sys
import time
sys.path.append(osp.join(sys.path[0], '..'))

import torch
from torch.utils import data

import models
from mmcv import Config
from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed)
from tools.apis.train import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector, detectors
from tools import utils
from mmdet.datasets import build_dataloader
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

import warnings
warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # retinanet
#    parser.add_argument('--config', default='./configs/fna_retinanet_fpn_retrain.py', help='train config file path')
#    parser.add_argument('--work_dir', default='./', type=str, help='the dir to save logs and models')
#    parser.add_argument('--data_path', default='../../datasets/coco/', type=str, help='the data path')
#    parser.add_argument('--job_name', type=str, default='', help='job name for output path')
#    parser.add_argument(
#        '--resume_from', help='the checkpoint file to resume from')
#    parser.add_argument(
#        '--validate',
#        action='store_true',
#        help='whether to evaluate the checkpoint during training')
#    parser.add_argument(
#        '--gpus',
#        type=int,
#        default=1,
#        help='number of gpus to use '
#        '(only applicable to non-distributed training)')
#    parser.add_argument('--seed', type=int, default=1, help='random seed')
#    parser.add_argument('--port', type=int, default=23333, help='random seed')
#    parser.add_argument(
#        '--launcher',
#        choices=['none', 'pytorch', 'slurm', 'mpi'],
#        default='pytorch',
#        help='job launcher')
#    parser.add_argument('--local_rank', type=int, default=0)
    
    # SSD Lite
    parser.add_argument('--model', default='ssdlite', help='model name') # temp
    parser.add_argument('--config', default='./configs/fna_ssdlite_retrain.py', help='train config file path')
    parser.add_argument('--work_dir', default='./output/', type=str, help='the dir to save logs and models')
    parser.add_argument('--data_path', default='../../datasets/coco/', type=str, help='the data path')
    parser.add_argument('--job_name', type=str, default='', help='job name for output path')
    parser.add_argument(
        '--resume_from', default='', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        default=True,
#        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--port', type=int, default=23333, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        if args.job_name is '':
            args.job_name = osp.join('output', args.model + '_param')
            # args.job_name = 'output'
        else:
            args.job_name = time.strftime("%Y%m%d-%H%M%S-") + args.job_name
        cfg.work_dir = osp.join(args.work_dir, args.job_name)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '%d' % args.port
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    # logger = get_root_logger(cfg.log_level)
    utils.create_work_dir(cfg.work_dir)
    logger = utils.get_root_logger(cfg.work_dir, cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Retrain configs: \n'+str(cfg))
    logger.info('Retrain args: \n'+str(args))

    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    
    utils.set_data_path(args.data_path, cfg.data)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if not hasattr(model, 'neck'):
        model.neck = None
    
    logger.info('Backbone net config: \n' + cfg.model.backbone.net_config)
    utils.get_network_madds(model.backbone, model.neck, model.bbox_head, 
                            cfg.image_size_madds, logger)

    if cfg.use_syncbn:
        model = utils.convert_sync_batchnorm(model)

    train_dataset = build_dataset(cfg.data.train)
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)

    logger.info('Backbone net config: \n' + cfg.model.backbone.net_config)
    utils.get_network_madds(model.backbone, model.neck, model.bbox_head, 
                            cfg.image_size_madds, logger)


if __name__ == '__main__':
    main()
