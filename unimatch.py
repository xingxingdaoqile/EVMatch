import argparse
import logging
import os
import pprint
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter, sec_to_hm_str
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--with_cp', action='store_true')#是否启用检查点机制


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0 #这段代码首先通过 init_log 函数初始化了一个名为 'global' 的日志记录器，并将其日志级别设置为 INFO。接着通过设置 logger.propagate = 0，关闭了日志消息的传播功能，因此日志消息不会传递给父记录器。

    rank, world_size = setup_distributed(port=args.port)#设置分布式训练环境，并返回两个值：rank 和 world_size
    # rank = 0（它是第几个进程）
    # world_size = 1（GPU的数量）

    if rank == 0:#这段代码用于判断当前进程是否为主进程（即 rank == 0）。
    #在分布式训练中，通常只有 rank == 0 的主进程执行某些全局操作，比如日志记录、模型保存等，而其他进程则专注于训练计算。
        all_args = {**cfg, **vars(args), 'ngpus': world_size}#这里创建了一个字典 all_args，用来合并配置文件 cfg 和命令行参数 args，并添加一个键 'ngpus'，它的值为 world_size（即 GPU 的数量）。**cfg 和 **vars(args) 是 Python 的字典解包语法，用于将 cfg 和 args 中的所有键值对合并到新的字典 all_args 中。vars(args) 将 args 对象（通常是通过 argparse 解析的命令行参数）转换为字典形式
        logger.info('{}\n'.format(pprint.pformat(all_args)))#这行代码用于将 all_args 中的内容格式化为字符串，并使用 logger.info 记录到日志中。pprint.pformat 是 Python 中的 pprint 模块的一个函数，用于将复杂的 Python 数据结构（如字典、列表）格式化为易读的字符串形式。这个步骤只会在主进程（rank == 0）中执行。
        
        writer = SummaryWriter(args.save_path)#这一行代码用于初始化一个 TensorBoard 的 SummaryWriter 实例。SummaryWriter 是 PyTorch 中用于将训练的指标（如损失、精度等）写入 TensorBoard 的工具。args.save_path 指定了将数据保存到的路径。
        
        os.makedirs(args.save_path, exist_ok=True)#这行代码用于创建保存模型或日志文件的目录。如果目录 args.save_path 不存在，则创建它；如果目录已经存在，exist_ok=True 确保不会引发错误。
    
    cudnn.enabled = True #启用 cuDNN 加速库，这是 PyTorch 默认的 GPU 加速选项，用于卷积操作。
    cudnn.benchmark = True #启用 cuDNN 的自动优化模式，在输入大小不变的情况下，cuDNN 会通过多次测试选择最佳的卷积算法。这有助于提高模型在固定输入尺寸上的性能。

    model = DeepLabV3Plus(cfg, args.with_cp)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    # 使用 SGD 优化器，分别为模型的骨干网络 backbone 和其他部分设置不同的学习率。这样可以对不同层的参数施加不同的学习率进行优化。

    if rank == 0: #统计并输出模型的参数总量（以百万为单位），只有主进程需要记录这些信息。
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    # 获取本地 GPU 编号，并将模型放入 GPU 中 (model.cuda())。
    # os.environ 是 Python 中用于存储系统环境变量的字典，os.environ["LOCAL_RANK"] 获取环境变量 LOCAL_RANK 的值，LOCAL_RANK 的值表示当前机器上使用的 GPU 的标识符（比如 0、1 等等）
    # torch.nn.SyncBatchNorm 用于分布式训练。convert_sync_batchnorm(model) 方法将模型中所有的普通 BatchNorm 层转换为同步批归一化层（SyncBatchNorm）。在多 GPU 训练时，SyncBatchNorm 能够跨不同 GPU 同步计算批归一化的均值和方差，从而提高模型在分布式环境下的训练效果
    # torch.nn.parallel.DistributedDataParallel（简称 DDP）是 PyTorch 用于多 GPU 分布式训练的模块，它能够有效地在多台机器或多个 GPU 上进行模型训练
    local_rank = int(os.environ["LOCAL_RANK"]) #获取当前进程使用的 GPU 的本地编号
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #将模型中的批归一化层替换为同步批归一化层，以适应分布式训练
    model.cuda()

    # broadcast_buffers=False 禁用模型中非参数缓冲区（比如 BatchNorm 层的均值和方差）的广播操作。广播操作会在每次前向传播时将主进程上的缓冲区复制到其他进程上
    # find_unused_parameters=False 禁用寻找未使用参数的功能。当模型的一些参数没有在前向传播中使用时，这个参数会影响梯度同步
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False) #将模型包装为分布式数据并行模型，以便在多个 GPU 上进行高效的并行训练

    #**cfg['criterion']['kwargs'] 表示将配置文件中的额外参数传递给 CrossEntropyLoss。这些参数可能包括 weight（类别权重）或 reduction（损失的计算方式）等
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM': #在线难例挖掘
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    #设置 reduction='none'，表示不进行任何聚合操作。返回的损失值会是与输入相同维度的张量，包含每个样本、每个像素的单独损失值
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank) #这意味着每个输入样本或像素点都会得到一个对应的损失值，而不会将整个批次的损失合并成一个单一的数值

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    start_time = 0
    start_iter = (epoch + 1) * len(trainloader_u)
    if rank == 0:
        start_time = time.time()
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                time_sofar = time.time() - start_time
                training_time_left = ((total_iters - start_iter) / (
                            iters - start_iter) - 1.0) * time_sofar if iters - start_iter > 0 else 0
                print_info = 'Iters: {:}, Total loss: {:.3f}, \n' + \
                    '                                    ' + \
                    'Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}, \n' + \
                    '                                    ' + \
                    'time elapsed: {}, time left: {}'
                logger.info(print_info.format(i, total_loss.avg,
                                              total_loss_x.avg,
                                              total_loss_s.avg,
                                              total_loss_w_fp.avg,
                                              total_mask_ratio.avg,
                                              sec_to_hm_str(time_sofar),
                                              sec_to_hm_str(
                                                  training_time_left)))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, rank)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            logger.info('***** Previous best ***** >>>> MeanIoU: {:.2f}\n'.format(previous_best))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
