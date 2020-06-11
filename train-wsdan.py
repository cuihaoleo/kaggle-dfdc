"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os
import time
import logging
import warnings
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import sys
from model_def import WSDAN
from dfdc_dataset import DfdcDataset
from wsdan_utils import CenterLoss, AverageMeter, TopKAccuracyMetric, batch_augment
import cv2
import importlib.util

import_spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(import_spec)
import_spec.loader.exec_module(config)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# GPU settings
assert torch.cuda.is_available()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1,))
crop_metric = TopKAccuracyMetric(topk=(1,))
drop_metric = TopKAccuracyMetric(topk=(1,))


def main_worker(local_rank, ngpus_per_node, args):
    if local_rank == 0:
        logging.basicConfig(
            filename=os.path.join(config.save_dir, config.log_name),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)
    warnings.filterwarnings("ignore")
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:22365', world_size=ngpus_per_node, rank=local_rank)
    torch.cuda.set_device(local_rank)
    train_dataset = DfdcDataset(phase='train', datapath=config.datapath, resize=config.image_size)
    validate_dataset = DfdcDataset(phase='val', datapath=config.datapath, resize=config.image_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    validate_sampler = torch.utils.data.distributed.DistributedSampler(validate_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, pin_memory=True, num_workers=config.workers)
    validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size, sampler=validate_sampler, pin_memory=True, num_workers=config.workers)

    num_classes = train_dataset.num_classes
    logs = {}
    start_epoch = 0
    net = WSDAN(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=config.pretrained)
    num_features = net.num_features
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    cross_entropy_loss = nn.CrossEntropyLoss().to(local_rank)
    center_loss = CenterLoss().to(local_rank)
    feature_center = torch.zeros(num_classes, config.num_attentions * num_features).to(local_rank)

    if config.ckpt:
        # Load ckpt and get state_dict
        loc = 'cuda:{}'.format(local_rank)
        checkpoint = torch.load(config.ckpt, map_location=loc)
        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])
        # Load weights
        state_dict = checkpoint['state_dict']
        net.module.load_state_dict(state_dict)
        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = F.normalize(checkpoint['feature_center'], dim=-1)

    ##################################
    # Optimizer, LR Scheduler
    ##################################
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
    #learning_rate =  config.learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    for epoch in range(start_epoch, config.epochs):
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']
        train_sampler.set_epoch(epoch)
        train_sampler.dataset.next_epoch()
        train(logs=logs,
              data_loader=train_loader,
              net=net, cross_entropy_loss=cross_entropy_loss, center_loss=center_loss,
              feature_center=feature_center,
              optimizer=optimizer, ngpus_per_node=ngpus_per_node, local_rank=local_rank)
        validate(logs=logs,
                 data_loader=validate_loader, cross_entropy_loss=cross_entropy_loss,
                 net=net, ngpus_per_node=ngpus_per_node, local_rank=local_rank)
        scheduler.step()
        feature_center*=2   #amplifiy feature centers, this keeps normalized feature center but change update rate. 
        if local_rank == 0:
            torch.save({
                'logs': logs,
                'state_dict': net.module.state_dict(),
                'feature_center': feature_center}, config.save_dir+'ckpt_%s.pth' % epoch)
        dist.barrier()


def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    ngpus_per_node = kwargs['ngpus_per_node']
    local_rank = kwargs['local_rank']
    cross_entropy_loss = kwargs['cross_entropy_loss']
    center_loss = kwargs['center_loss']
    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    for i, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        X = X.to(local_rank, non_blocking=True)
        y = y.to(local_rank, non_blocking=True)
        y_pred_raw, feature_matrix, attention_map = net(X, dropout=True)
        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)
        dist.all_reduce(feature_center, op=dist.ReduceOp.SUM)
        feature_center /= ngpus_per_node
        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

        # crop images forward
        y_pred_crop, _, _ = net(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.4, 0.7))

        # drop images forward
        y_pred_drop, _, _ = net(drop_images)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) + \
            cross_entropy_loss(y_pred_crop, y) / 3. + \
            cross_entropy_loss(y_pred_drop, y) / 2. + \
            center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_crop, y)
            epoch_drop_acc = drop_metric(y_pred_drop, y)
        # end of this batch
    epoch_loss = torch.tensor(epoch_loss).cuda()
    epoch_raw_acc = torch.tensor(epoch_raw_acc).cuda()
    epoch_crop_acc = torch.tensor(epoch_crop_acc).cuda()
    epoch_drop_acc = torch.tensor(epoch_drop_acc).cuda()
    dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(epoch_raw_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(epoch_crop_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(epoch_drop_acc, op=dist.ReduceOp.SUM)
    epoch_loss = epoch_loss.item()/ngpus_per_node
    epoch_raw_acc = epoch_raw_acc.cpu().numpy()/ngpus_per_node
    epoch_crop_acc = epoch_crop_acc.cpu().numpy()/ngpus_per_node
    epoch_drop_acc = epoch_drop_acc.cpu().numpy()/ngpus_per_node
    batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}), Crop Acc ({:.2f}), Drop Acc ({:.2f})'.format(
        epoch_loss, epoch_raw_acc[0],
        epoch_crop_acc[0], epoch_drop_acc[0])
    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    if local_rank == 0:
        logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    ngpus_per_node = kwargs['ngpus_per_node']
    local_rank = kwargs['local_rank']
    cross_entropy_loss = kwargs['cross_entropy_loss']
    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.to(local_rank, non_blocking=True)
            y = y.to(local_rank, non_blocking=True)

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, _, attention_map = net(X)

            ##################################
            # Object Localization and Refinement
            ##################################
            crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = net(crop_images)

            ##################################
            # Final prediction
            ##################################
            y_pred = (y_pred_raw + y_pred_crop) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)

    # end of validation
        epoch_loss = torch.tensor(epoch_loss).cuda()
        epoch_acc = torch.tensor(epoch_acc).cuda()
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_acc, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss.item()/ngpus_per_node
        epoch_acc = epoch_acc.cpu().numpy()/ngpus_per_node
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f})'.format(epoch_loss, epoch_acc[0])

    # write log for this epoch
    if local_rank == 0:
        logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
        logging.info('')


if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, ''))
