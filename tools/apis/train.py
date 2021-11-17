from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import HOOKS, Runner, DistSamplerSeedHook, build_optimizer, build_runner
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import DistOptimizerHook, DistEvalHook
# from mmdet.core import (CocoDistEvalRecallHook, CocoDistEvalmAPHook,
#                         DistEvalmAPHook, DistOptimizerHook)
from mmdet.datasets import build_dataloader, replace_ImageToTensor, build_dataset
from mmdet.models import RPN
from mmdet.utils import get_root_logger
from mmcv.utils import build_from_cfg

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None,
                   local_rank=0):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate, logger=logger, local_rank=local_rank)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate, logger=logger, )


# adopted from mmdet train_detector function
def _dist_train(model, dataset, cfg, validate=False, logger=None, local_rank=0):
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True,
            runner_type=runner_type)
    ]

    # put model on gpus
    # model = MMDistributedDataParallel(model.cuda())
    # model = MMDistributedDataParallel(model.cuda(), find_unused_parameters=True,
    #                                 device_ids=[local_rank], output_device=[local_rank])
    # model = MMDistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=[local_rank])
    model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False,)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger))
    
    # build runner
    # optimizer = build_optimizer(model, cfg.optimizer)
    # # remove batch_processor to utilize the default "train_step" and "val_step" of BaseDetector class of mmdet
    # runner = Runner(model, batch_processor=None, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger)
    
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=True,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    
    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
            
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    runner.run(data_loaders, cfg.workflow)


def _non_dist_train(model, dataset, cfg, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    # runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir, cfg.log_level)
    runner = Runner(model, batch_processor=None, optimizer=cfg.optimizer, work_dir=cfg.work_dir, logger=cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
