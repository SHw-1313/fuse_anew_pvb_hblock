#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import torch.distributed
from tqdm import tqdm
import wandb
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.logger import print_log
from utils.ema import EMA
from utils.fusion_training import (
    configure_fusion_parameters,
    fusion_gradient_norms,
    fusion_parameter_groups,
)


def replace_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.nan_to_num(param.grad.data)


########### Import your packages below ##########
class TrainConfig:
    def __init__(self, save_dir, lr, max_epoch,
                 metric_min_better=True, warmup=1000, patience=3,
                 grad_clip=None, save_topk=-1,  # -1 for save all
                 **kwargs):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.metric_min_better = metric_min_better
        self.warmup = warmup
        self.patience = patience
        self.grad_clip = grad_clip
        self.save_topk = save_topk
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config):
        self.model = model
        self.config = config
        self.fusion_training_info = configure_fusion_parameters(
            self.model,
            stage=getattr(self.config, 'fusion_stage', 'standard'),
            unfreeze_ept_layers=getattr(self.config, 'unfreeze_ept_layers', 2),
            source_keys=getattr(self.model, '_source_checkpoint_keys', None),
        )
        self.ema = self.get_ema()
        self.optimizer = self.get_optimizer()
        warmup_config = self.get_warmup_scheduler(self.optimizer)
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.warmup_scheduler = warmup_config['scheduler']
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        # log
        self.version = self._get_version()
        self.config.save_dir = os.path.join(self.config.save_dir, f'version_{self.version}')
        self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')
        self.writer = None  # initialize right before training
        self.writer_buffer = {}

        # training process recording
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []  # smaller index means better ckpt
        self.patience = self.config.patience

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        version, pattern = -1, r'version_(\d+)'
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return version + 1

    def _train_epoch(self, device):
        if self.epoch > 0 and hasattr(self.train_loader.dataset, 'update_epoch'):
            self.train_loader.dataset.update_epoch()
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            try:
                self.train_loader.sampler.set_epoch(self.epoch)
            except BaseException:
                self.train_loader.batch_sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            try:
                with self._autocast_context(device):
                    loss = self.train_step(batch, self.global_step)
                if torch.isnan(loss):
                    print_log('encounter NaN loss, skip batch', level='WARN')
                    continue
                # torch.autograd.set_detect_anomaly(True)
                loss.backward()
                if getattr(self.model, 'fusion_mode', 'off') == 'anew_block':
                    for name, value in fusion_gradient_norms(self.model).items():
                        self.log(f'Fusion/{name}', value, self.global_step)
                replace_nan_gradients(self.model)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print_log('CUDA out of memory, skip batch', level='WARN')
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update()
            if hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(loss=loss.item(), version=self.version)
            self.global_step += 1
            if self.global_step < self.config.warmup:
                self.warmup_scheduler.step()
            elif self.sched_freq == 'batch':
                self.scheduler.step()
        if self.global_step >= self.config.warmup and self.sched_freq == 'epoch':
            self.scheduler.step()

    def _valid_epoch(self, device):
        # if self.epoch > 0 and hasattr(self.valid_loader.dataset, 'update_epoch'):
        #     self.valid_loader.dataset.update_epoch()
        metric_arr = []
        self.model.eval()

        if self.valid_loader.sampler is not None and self.local_rank != -1:  # distributed
            try:
                self.valid_loader.sampler.set_epoch(self.epoch)
            except BaseException:
                self.valid_loader.batch_sampler.set_epoch(self.epoch)

        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                with self._autocast_context(device):
                    metric = self.valid_step(batch, self.valid_global_step)
                if torch.isnan(metric):
                    print_log('encounter NaN metric, skip batch', level='WARN')
                    continue
                if torch.cuda.is_available() and torch.distributed.is_initialized():
                    metric = metric.unsqueeze(0)
                    world_size = torch.distributed.get_world_size()
                    gathered = [torch.zeros_like(metric) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered, metric)
                    if self._is_main_proc():
                        metric_arr.extend([m.cpu().item() for m in gathered])
                else:
                    metric_arr.append(metric.cpu().item())
                if hasattr(t_iter, 'set_postfix'):
                    t_iter.set_postfix(metric=metric.item())
                self.valid_global_step += 1
        
        self.model.train()

        # calculate valid metric in main proc, and broadcast to all procs
        if self._is_main_proc():
            valid_metric = float(np.nanmean(metric_arr))
        else:
            valid_metric = 0.0
        
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            valid_metric_ts = torch.tensor([valid_metric], dtype=torch.float, device=device)
            torch.distributed.broadcast(valid_metric_ts, src=0)
            valid_metric = valid_metric_ts.cpu().item()
        
        if self._is_main_proc():
            if self._metric_better(valid_metric):
                self.patience = self.config.patience
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                checkpoint = {
                    'format_version': 1,
                    'model_state_dict': module_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict() \
                        if self.warmup_scheduler is not None else None,
                    'scheduler_state_dict': self.scheduler.state_dict() \
                        if self.scheduler is not None else None,
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                    'valid_global_step': self.valid_global_step,
                    'best_metric': valid_metric,
                    'patience': self.patience,
                }
                if self.ema is not None:
                    checkpoint['ema_model_state_dict'] = self.ema.ema_model.state_dict()
                    checkpoint['ema_initted'] = self.ema.initted.detach().cpu()
                    checkpoint['ema_step'] = self.ema.step.detach().cpu()
                torch.save(checkpoint, save_path)
                self._maintain_topk_checkpoint(valid_metric, save_path)
            else:
                self.patience -= 1
            self.last_valid_metric = valid_metric

            # Reduce on Plateau
            if self.global_step >= self.config.warmup and self.sched_freq == 'val_epoch':
                self.scheduler.step(valid_metric)

            # write valid_metric
            for name in self.writer_buffer:
                value = np.nanmean(self.writer_buffer[name])
                self.log(name, value, self.epoch, val=True)
            self.writer_buffer = {}
        
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            lrs_ts = torch.tensor(lrs, dtype=torch.float, device=device)
            torch.distributed.broadcast(lrs_ts, src=0)
            for lr, group in zip(lrs_ts.cpu().tolist(), self.optimizer.param_groups):
                group['lr'] = lr

    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def _autocast_context(self, device):
        enabled = bool(getattr(self.config, 'bf16_autocast', False))
        if device.type not in {'cuda', 'cpu'}:
            enabled = False
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)

    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        topk = self.config.save_topk
        if self.config.metric_min_better:
            better = lambda a, b: a < b
        else:
            better = lambda a, b: a > b
        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break
        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))

        # maintain topk
        if topk > 0:
            while len(self.topk_ckpt_map) > topk:
                last_ckpt_path = self.topk_ckpt_map[-1][1]
                os.remove(last_ckpt_path)
                self.topk_ckpt_map.pop()

        # save map
        topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        with open(topk_map_path, 'w') as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f'{metric}: {path}\n')

    def train(self, device_ids, local_rank):
        # set local rank
        self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            # self.writer = SummaryWriter(self.config.save_dir)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(os.path.join(self.config.save_dir, 'train_config.json'), 'w') as fout:
                json.dump(self.config.__dict__, fout)
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        self.model.to(device)
        if self.ema:
            self.ema.to(device)
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            print_log(f'training on {device_ids}')
        for _ in range(self.config.max_epoch):
            print_log(f'epoch{self.epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            print_log(f'validating ...') if self._is_main_proc() else 1
            self._valid_epoch(device)
            self.epoch += 1
            if self.patience <= 0:
                break

    def restore_resume_state(self, metadata):
        """Restore non-model state after the trainer has built its optimizers."""

        if metadata.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(metadata['optimizer_state_dict'])
        if self.warmup_scheduler is not None and metadata.get('warmup_scheduler_state_dict') is not None:
            self.warmup_scheduler.load_state_dict(metadata['warmup_scheduler_state_dict'])
        if self.scheduler is not None and metadata.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(metadata['scheduler_state_dict'])
        if self.ema is not None and metadata.get('ema_model_state_dict') is not None:
            self.ema.ema_model.load_state_dict(metadata['ema_model_state_dict'])
            if metadata.get('ema_initted') is not None:
                self.ema.initted.copy_(metadata['ema_initted'].to(self.ema.initted.device))
            if metadata.get('ema_step') is not None:
                self.ema.step.copy_(metadata['ema_step'].to(self.ema.step.device))
        self.epoch = int(metadata.get('epoch', self.epoch))
        self.global_step = int(metadata.get('global_step', self.global_step))
        self.valid_global_step = int(metadata.get('valid_global_step', self.valid_global_step))
        self.last_valid_metric = metadata.get('best_metric', self.last_valid_metric)
        self.patience = int(metadata.get('patience', self.patience))

    def log(self, name, value, step, val=False, accumulation=False):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if accumulation:
                if name not in self.writer_buffer:
                    self.writer_buffer[name] = []
                self.writer_buffer[name].append(value)
            else:
                step_name = "train_step" if not val else "valid_step"
                wandb.log({name: value, step_name: step})
                # self.writer.add_scalar(name, value, step)

    ########## Overload these functions below ##########
    # define model wrapper
    def get_ema(self):
        ema = None
        if hasattr(self.config, 'ema'):
            ema = EMA(self.model, **self.config.ema)
        return ema 

    # define optimizer
    def get_optimizer(self):
        if getattr(self.model, 'fusion_mode', 'off') == 'anew_block':
            groups = fusion_parameter_groups(
                self.model,
                pvb_lr=getattr(self.config, 'pvb_lr', self.config.lr),
                anew_lr=getattr(self.config, 'anew_lr', self.config.lr),
                projector_lr=getattr(self.config, 'projector_lr', self.config.lr),
            )
            if not groups:
                raise ValueError('Anew fusion has no trainable parameters after stage configuration')
            return torch.optim.Adam(groups, eps=1e-8)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=1e-8)
        return optimizer

    def get_warmup_scheduler(self, optimizer):
        lam = lambda step: float(step + 1) / float(self.config.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {
            'scheduler': scheduler,
            # 'frequency': 'batch'
        }

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1.0e-7)
        return {
            'scheduler': scheduler,
            'frequency': 'val_epoch'  # or batch, epoch
        }

    # train step, note that batch should be dict/list/tuple/instance. Objects with .to(device) attribute will be automatically moved to the same device as the model
    def train_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        lr = self.optimizer.param_groups[0]['lr']
        self.log('lr', lr, batch_idx)
        return loss

    # validation step
    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss
