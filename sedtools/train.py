# coding:utf-8
import os
import shutil
import sys

project_dir = os.getenv('PROJ_DIR')
os.chdir(project_dir)
sys.path.append(project_dir)

import torch
from torch.backends import cudnn
import numpy as np
import random
from argparse import ArgumentParser

import time

from utils.loss import criterion_all
from torch.utils.data import DataLoader
from runtime_utils import freeze_batchnorm
from tqdm import tqdm
from sedtools.components.buildExtractor import build_extractor
from sedtools.components.buildModel import build_model


def before_train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(os.path.join(args.save_dir, "visual")):
        os.makedirs(os.path.join(args.save_dir, "visual"))

    with open(args.save_dir + '/args.txt', "w") as fp:
        fp.write(str(args))


def build_dataset(args):
    if args.dataset == "sbd":
        from utils.sbd_datasets import SBDTrain, SBDVal
        dataset_train = SBDTrain(args.root_dir, args.flist_train, crop_size=args.crop_size)
        dataset_val = SBDVal(args.root_dir, args.flist_val, crop_size=args.crop_size)

    elif args.dataset == "cityscapes":
        from utils.city_datasets import CityTrain, CityVal
        dataset_train = CityTrain(args.root_dir, args.flist_train, crop_size=args.crop_size)
        dataset_val = CityVal(args.root_dir, args.flist_val, crop_size=args.crop_size)

    elif args.dataset == "ade20k":
        from utils.ade20k.ade20k_datasets import ADE20kTrain, ADE20kVal
        dataset_train = ADE20kTrain(args.root_dir, args.flist_train, crop_size=args.crop_size)
        dataset_val = ADE20kVal(args.root_dir, args.flist_val, crop_size=args.crop_size)

    else:
        raise ValueError(f"unknown dataset, {args.dataset}")

    return dataset_train, dataset_val


def build_optimizer(args, extractor, model):
    if args.backbone == "resnet101":
        from runtime_utils import model_parameters_resnet101 as model_parameters
        parameters_id = model_parameters(extractor, model)
        group_params = [
            # {'params': parameters_id['backbone.parameters'], 'lr': args.base_lr * 1, 'weight_decay': 0.01},
            {'params': parameters_id['res_conv.weight'], 'lr': args.base_lr * 1, 'weight_decay': 0.01},
            {'params': parameters_id['res_bn.weight'], 'lr': 0, 'weight_decay': 0},
            {'params': parameters_id['res_bn.bias'], 'lr': 0, 'weight_decay': 0},
            {'params': parameters_id['score.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['score.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01}
        ]
    elif args.backbone in ["vit_det"]:
        from runtime_utils import model_parameters_vit as model_parameters
        parameters_id = model_parameters(extractor, model)
        group_params = [
            # {'params': parameters_id['adaptor.parameters'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['score.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['score.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01}
        ]
        if not args.freeze_vit:
            group_params.append(
                {'params': parameters_id['vit.parameters'], 'lr': args.base_lr * 1, 'weight_decay': 0.01},
            )
    elif "_adapter_" in args.backbone or 'rein' in args.backbone:
        from runtime_utils import model_parameters_vit_adapter as model_parameters
        parameters_id = model_parameters(extractor, model)
        group_params = [
            {'params': parameters_id['adaptor.parameters'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['score.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['score.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01}
        ]
        if not args.freeze_vit:
            print("finetuning vit parameters")
            group_params.append(
                {'params': parameters_id['vit.parameters'], 'lr': args.base_lr * 1, 'weight_decay': 0.01},
            )
    else:
        from runtime_utils import model_parameters
        parameters_id = model_parameters(extractor, model)
        group_params = [
            {'params': parameters_id['backbone.parameters'], 'lr': args.base_lr * 1, 'weight_decay': 0.01},
            # {'params': parameters_id['res_conv.weight'], 'lr': args.base_lr * 1, 'weight_decay': 0.01},
            # {'params': parameters_id['res_bn.weight'], 'lr': 0, 'weight_decay': 0},
            # {'params': parameters_id['res_bn.bias'], 'lr': 0, 'weight_decay': 0},
            {'params': parameters_id['score.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['score.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.weight'], 'lr': args.base_lr * 5, 'weight_decay': 0.01},
            {'params': parameters_id['br_conv.bias'], 'lr': args.base_lr * 5, 'weight_decay': 0.01}
        ]
    from torch.optim import AdamW
    optimizer = AdamW(
        params=group_params,
        lr=args.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )


def build_optimizer_and_scheduler(args, extractor, model, aux_model=None):
    group_params = list()
    if 'swin' in args.backbone:
        from optim import add_swin_params
        add_swin_params(
            group_params, extractor, base_lr=args.base_lr, lr_scale=1, weight_decay=0.01
        )
    else:
        from optim import add_vit_layer_decay_params
        add_vit_layer_decay_params(
            group_params, extractor, base_lr=args.base_lr, weight_decay=0.01
        )

    from torch.optim import AdamW
    optimizer = AdamW(
        params=group_params,
        lr=args.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    from optim import LrSchedulerWithWarmup
    scheduler = LrSchedulerWithWarmup(
        optimizer,
        max_iters=args.num_iters,
        base_lr=args.base_lr,
        warmup_steps=0.0,
        eta_min=1e-8,
        anneal_strategy='poly'
    )

    return optimizer, scheduler


def get_instances(gt_sed0, gt_seg0, ignore_label=255):
    gt_sed = gt_sed0.detach().clone().numpy()
    gt_seg = gt_seg0.detach().clone().numpy()

    classes = np.unique(gt_seg)
    classes = classes[classes != ignore_label]
    classes = classes[classes != 20]
    # classes = classes[classes != 150]
    gt_classes = torch.tensor(classes, dtype=torch.int64)

    mask_sed = []
    mask_seg = []

    ignore_mask_seg = gt_seg == ignore_label
    ignore_mask_sed = gt_sed == ignore_label

    for c in classes:
        gt_seg_mask = gt_seg == c
        gt_seg_mask[ignore_mask_seg] = 0

        gt_sed_mask = gt_sed[c]
        gt_sed_mask[ignore_mask_sed[c]] = 0

        mask_seg.append(gt_seg_mask)
        mask_sed.append(gt_sed_mask.astype(bool))  # 注意这里的格式区别

    if len(classes) == 0:
        h, w = gt_seg.shape
        gt_mask_seg = torch.zeros((0, h, w), dtype=torch.bool)
        ignore_mask_seg = torch.zeros((0, h, w), dtype=torch.bool)
        gt_mask_sed = torch.zeros((0, h, w), dtype=torch.bool)
        ignore_mask_sed = torch.zeros((0, h, w), dtype=torch.bool)

    else:
        gt_mask_seg = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_seg]).to(torch.bool)
        gt_mask_sed = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_sed]).to(torch.bool)
        ignore_mask_seg = torch.as_tensor(ignore_mask_seg[None], dtype=torch.bool).repeat([len(classes), 1, 1])
        ignore_mask_sed = torch.as_tensor(ignore_mask_sed[classes], dtype=torch.bool)
    instances = {"labels": gt_classes.to(gt_seg0.device),
                 "masks_sed": gt_mask_sed.to(gt_seg0.device),
                 "masks_seg": gt_mask_seg.to(gt_seg0.device),
                 "seg_ignore_masks": ignore_mask_seg.to(gt_seg0.device),
                 "sed_ignore_masks": ignore_mask_sed.to(gt_seg0.device)
                 }
    return instances


def data_process_mask_cls(args, labels, gt_seg, gt_edge, image_info):
    if args.model_name in ["aca_mask_cls", 'only_mask_cls', 'only_mask_cls2']:
        instances = []
        for i in range(labels.shape[0]):
            instances.append(get_instances(labels[i], gt_seg[i]))

        targets = {
            "gt_seg": gt_seg,
            "gt_edge": gt_edge,
            "gt_sed": labels,
            "instances": instances,
            "im_info": image_info
        }
        return {'targets': targets}
    elif args.model_name in ["aca", 'ahsi']:
        return {
            'gt_sed': labels,
            'gt_seg': gt_seg,
            'gt_edge': gt_edge
        }
    elif args.model_name in ['upernet', 'bimla']:
        return {
            'gt_sed': labels
        }


def lr_scheduler(optimizer, base_lr, cur_iter, max_iter, adjust_learning_rate):
    cur_lr = base_lr * pow(1 - (1.0 * cur_iter / max_iter), 0.9)
    adjust_learning_rate(optimizer, cur_lr)
    return cur_lr


def run_forward(args, images, extractor, model, aux_model=None):
    if args.backbone == "resnet101":
        res1, res2, res3, res4, res5 = extractor(images)
    elif 'vit_bimla' in args.backbone:
        feats = extractor(images)
    else:
        res1 = None
        res2, res3, res4, res5 = extractor(images)
    if 'vit_bimla' in args.backbone:
        results = model(feats)
    else:
        results = model(images, res1, res2, res3, res4, res5)
    if aux_model is not None:
        if isinstance(aux_model, torch.nn.ModuleList) and 'vit_bimla' in args.backbone:
            aux_results = list()
            for aux_model_ in aux_model:
                aux_result = aux_model_(feats)
                aux_results.append(aux_result)
        else:
            aux_results = aux_model(images, res1, res2, res3, res4, res5)
        return (results[0], aux_results) if isinstance(results, tuple) else (results, aux_results)
    return results


def build_criterion(args):
    if args.model_name in ["ahsi"]:
        from utils.loss import SBDAutoCrossEntropyLoss, Seg_CrossEntropy, Edge_bce
        weights = torch.Tensor(np.zeros((args.batch_size, args.nclasses, args.crop_size, args.crop_size))).to(
            args.device)
        criterion = torch.nn.DataParallel(SBDAutoCrossEntropyLoss(weights, pred_key=None, tgt_key=None)).to(args.device)

        weights_val = torch.Tensor(np.zeros((1, args.nclasses, args.crop_size, args.crop_size))).to(args.device)
        criterion_val = torch.nn.DataParallel(SBDAutoCrossEntropyLoss(weights_val, pred_key=None, tgt_key=None)).to(
            args.device)

        criterion_seg = torch.nn.DataParallel(Seg_CrossEntropy(pred_key=None, tgt_key=None)).to(args.device)

        weights_edge = torch.Tensor(np.zeros((args.batch_size, 1, args.crop_size, args.crop_size))).to(args.device)
        criterion_edge = torch.nn.DataParallel(Edge_bce(weights_edge, pred_key=None, tgt_key=None)).to(args.device)

        weights_edge_val = torch.Tensor(np.zeros((1, 1, args.crop_size, args.crop_size))).to(args.device)
        criterion_edge_val = torch.nn.DataParallel(Edge_bce(weights_edge_val, pred_key=None, tgt_key=None)).to(
            args.device)

        def run_criterion(results, gt_sed=None, gt_seg=None, gt_edge=None, cur_iter=None, max_iter=None, targets=None,
                          is_training=True):
            score, pred_seg, pred_edge = results
            if is_training:
                criterion1 = criterion
                criterion2 = criterion_edge
            else:
                criterion1 = criterion_val
                criterion2 = criterion_edge_val
            loss = criterion1(score, gt_sed, cur_iter, max_iter)
            loss_seg = criterion_seg(pred_seg, gt_seg)
            loss_edge = criterion2(pred_edge, gt_edge)
            loss_all = 1.0 * loss + 0.1 * loss_seg + 0.015 * loss_edge
            return loss_all, (loss, loss_seg, loss_edge)

    else:
        raise ValueError(f"Error: unknown backbone {args.backbone}")

    return run_criterion


def build_recoder(args):
    if args.model_name in ["ahsi"]:
        from aca_tools import Recorder

    else:
        raise ValueError(f"Error: unknown model_name when build_recorder {args.model_name}")
    return Recorder(args)


def build_visualizer(args):
    if args.model_name in ["ahsi"]:
        from aca_tools import visualize
    else:
        raise ValueError(f"Error: unknown model_name when build visualizer {args.model_name}")
    return visualize


def val_loop(args, extractor, model, dataset, run_criterion, recorder, cur_iter, max_iter, aux_model=None):
    extractor.eval()
    model.eval()
    recorder.reset_val_recs()
    with torch.no_grad():
        for i in tqdm(range(args.iters_val), total=args.iters_val):
            images, labels, gt_seg, gt_edge = dataset.train_data(i)
            targets = data_process_mask_cls(args, labels, gt_seg, gt_edge, None)
            images = images.to(args.device)
            results = run_forward(args, images, extractor, model, aux_model=aux_model)
            loss_all, loss_items = run_criterion(results, cur_iter=cur_iter, max_iter=max_iter, is_training=False,
                                                 **targets)
            if args.model_name not in ['upernet', 'bimla']:
                recorder.update(loss_all, loss_items, is_training=False)
            #     recorder.seg_metric_update(results, gt_seg)
        recorder.logout(cur_iter, max_iter, is_training=False)
    extractor.train()
    model.train()


def save_state(args, extractor, model, optimizer, cur_iter, aux_model=None):
    print("----- SAVE - ITER", cur_iter, "-----")
    filename = "%s/model-%4d.pth" % (args.save_dir, cur_iter)
    to_save = {
        'cur_iter': cur_iter,
        'extractor': extractor.state_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if aux_model is not None:
        to_save['aux_model'] = aux_model.state_dict()
    torch.save(to_save, filename)
    print("save: %s (Iter: %d)" % (filename, cur_iter))


def train(args):
    extractor, feat_chans = build_extractor(args)

    if args.aux_head is None:
        aux_model = None
        model = build_model(args, feat_chans)
    else:
        model, aux_model = build_model(args, feat_chans)

    dataset_train, dataset_val = build_dataset(args)
    loader_train = DataLoader(dataset_train, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    run_criterion = build_criterion(args)

    recorder = build_recoder(args)
    visualizer = build_visualizer(args)

    model_txt_path = args.save_dir + "/model.txt"
    with open(model_txt_path, 'w') as fp:
        fp.write(str(model))

    # optimizer
    # optimizer = build_optimizer(args, extractor, model)

    optimizer, scheduler = build_optimizer_and_scheduler(args, extractor, model, aux_model=aux_model)
    cur_iter = 0

    if args.pre_model and not args.save_model:
        filenameCheckpoint = args.pre_model
        assert os.path.exists(
            filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        if args.backbone == "swin_base" or args.backbone == "swin_small" or args.backbone == 'swin_tiny' or args.backbone == 'convnext':
            from runtime_utils import ckpt_pre_suffix
            ckpt = ckpt_pre_suffix(checkpoint['model'], model_prefix='module')
            extractor.load_state_dict(ckpt, strict=False)
            print("=> Loaded pre_model for backbone!")
        elif args.backbone == "resnet101":
            extractor.load_state_dict(checkpoint)
            print("=> Loaded pre_model for backbone!")
        else:
            raise ValueError("Unknown backbone when loading ckpt")

    if args.save_model:
        filenameCheckpoint = args.save_model
        assert os.path.exists(
            filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        extractor.load_state_dict(checkpoint['extractor'])
        model.load_state_dict(checkpoint['model'], strict=False)
        save_iter = 0

    extractor.train()
    model.train()
    freeze_batchnorm(extractor)
    max_iter = args.num_iters
    while cur_iter <= max_iter:
        for _, (images, labels, gt_seg, gt_edge, image_info) in enumerate(loader_train):
            if args.save_model is not None and cur_iter < save_iter:
                print("skiping iters: ", cur_iter, " target_iter: ", save_iter)
                cur_iter += 1
                continue
            start_time = time.time()

            targets = data_process_mask_cls(args, labels, gt_seg, gt_edge, image_info)

            cur_iter += 1
            if cur_iter > max_iter:
                return

            images = images.to(args.device)
            results = run_forward(args, images, extractor, model, aux_model=aux_model)

            loss_all, loss_items = run_criterion(results, cur_iter=cur_iter, max_iter=max_iter, **targets)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            cur_lr = scheduler.step(cur_iter)

            recorder.update(loss_all, loss_items, start_time)
            if cur_iter % args.iters_loss == 0:
                recorder.logout(cur_iter, cur_lr)

            if cur_iter % args.iters_visual == 0:
                visualizer(args, images, labels, cur_iter, results)

            if cur_iter % args.iters_save == 0:
                val_loop(args, extractor, model, dataset_val, run_criterion, recorder, cur_iter, max_iter,
                         aux_model=aux_model)
                save_state(args, extractor, model, optimizer, cur_iter, aux_model=aux_model)


def main(args):
    print("========== BEFORE TRAIN ===========")
    before_train(args)
    print("========== NORMAL TRAINING  START===========")
    train(args)
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    SEED = 2
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cudnn.enabled = True
    cudnn.benchmark = True

    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='datasets/cityscapes/data_proc')  # cityscapes
    parser.add_argument('--flist_train', default='datasets/cityscapes/data_proc/train.txt')
    parser.add_argument('--flist_val', default='datasets/cityscapes/data_proc/val.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--num-iters', type=int, default=320000)  # cityscapes
    parser.add_argument('--nclasses', type=int, default=19)  # cityscapes
    parser.add_argument('--base-lr', type=float, default=2e-5)  # sbd 1e-5 学习率要合适
    parser.add_argument('--crop-size', type=int, default=512)  # cityscapes
    parser.add_argument('--batch-size', type=int, default=2)  # cityscapes
    parser.add_argument('--iters-average', type=int, default=40)  # 100 仅与平均损失有关
    parser.add_argument('--iters-loss', type=int, default=50)  # 100
    parser.add_argument('--iters-val', type=int, default=500)  # 100
    parser.add_argument('--iters-visual', type=int, default=500)  # 1000
    parser.add_argument('--iters-save', type=int, default=5000)
    parser.add_argument('--save-dir', default='save_disk/debug')
    parser.add_argument('--pre-model', default='ckpt/convnextv2_base_22k_384_ema.pt')
    parser.add_argument('--save-model', default=None)
    parser.add_argument("--n_queries", type=int, default=40)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--num_aux_layers", type=int, default=6)
    parser.add_argument("--loss_seg", type=float, default=0.1)
    parser.add_argument("--loss_edge", type=float, default=0.015)
    parser.add_argument("--cls_weight", type=float, default=5.0)
    parser.add_argument("--sed_weight", type=float, default=1.0)
    parser.add_argument("--dice_sed_weight", type=float, default=1.0)
    parser.add_argument("--no_object_weight", type=float, default=0.1)
    parser.add_argument("--no_object_bce_weight", type=float, default=0.1)
    parser.add_argument("--loss_queries_const", type=float, default=0.1)
    parser.add_argument("--backbone", type=str, default="convnext")
    parser.add_argument("--vit-name", type=str, default="dinov2_large")
    parser.add_argument("--freeze-vit", action='store_true', default=False)
    parser.add_argument("--freeze-spm", action='store_true', default=False)
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--vit-top-feats", action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default="ade20k", choices=['sbd', 'cityscapes', 'ade20k'])
    parser.add_argument("--model-name", type=str, default="aca_mask_cls")
    parser.add_argument("--aux-head", type=str, default=None)
    parser.add_argument("--spm-name", type=str, default=None)
    parser.add_argument("--mask-attn-t", type=float, default=0.4)
    main(parser.parse_args())
