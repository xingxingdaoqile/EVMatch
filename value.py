import argparse
import logging
import os
import pprint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.deeplabv3pluss import DLSideout
from util.classes import CLASSES
from util.utils import init_log, intersectionAndUnion, AverageMeter

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)


def evaluate(model, loader, mode, cfg, rank):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    pbar=None
    if rank == 0:
        pbar = tqdm(total=len(loader), desc='Value', leave=True, ncols=100, unit='bc', unit_scale=True)

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

            if rank == 0:
                pbar.update(1)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # rank, world_size = setup_distributed(port=args.port)
    rank = 0
    world_size = 1

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # model = DeepLabV3Plus(cfg, False)
    model = DLSideout(cfg, False)

    local_rank = 0
    model.cuda(local_rank)

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=None)

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        filtered_dict_enc = {k[7:]: v for k, v in checkpoint['model'].items()}
        model.load_state_dict(filtered_dict_enc, strict=True)
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint Miou is %f\n' % previous_best)

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    # eval_mode = 'original'
    mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, rank)

    if rank == 0:
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))


if __name__ == '__main__':
    main()
