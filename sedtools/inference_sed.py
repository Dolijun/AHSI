import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import re
import scipy.io as sci
import tqdm
import math
from components.buildExtractor import build_extractor

class ActivationWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activation = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )

    def save_activation(self, module, input, output):
        self.activation.append(output)

    def __call__(self, inputs):

        ## todo: model inference
        self.activation = []
        results = self.model(inputs)
        return results, self.activation

    def release(self):
        for handle in self.handles:
            handle.remove()


def get_visual_colors(dataset):
    # RGB !!!
    if dataset == 'cityscapes':
        colors = [[128, 64, 128],
                  [244, 35, 232],
                  [70, 70, 70],
                  [102, 102, 156],
                  [190, 153, 153],
                  [153, 153, 153],
                  [250, 170, 30],
                  [220, 220, 0],
                  [107, 142, 35],
                  [152, 251, 152],
                  [70, 130, 180],
                  [220, 20, 60],
                  [255, 0, 0],
                  [0, 0, 142],
                  [0, 0, 70],
                  [0, 60, 100],
                  [0, 80, 100],
                  [0, 0, 230],
                  [119, 11, 32]]
    elif dataset == "ade20k":
        from utils.ade20k.ade20k_stuff import stuff_colors
        colors = stuff_colors
    else:
        assert dataset == 'sbd'
        colors = [[128, 0, 0],
                  [0, 128, 0],
                  [128, 128, 0],
                  [0, 0, 128],
                  [128, 0, 128],
                  [0, 128, 128],
                  [128, 128, 128],
                  [64, 0, 0],
                  [192, 0, 0],
                  [64, 128, 0],
                  [192, 128, 0],
                  [64, 0, 128],
                  [192, 0, 128],
                  [64, 128, 128],
                  [192, 128, 128],
                  [0, 64, 0],
                  [128, 64, 0],
                  [0, 192, 0],
                  [128, 192, 0],
                  [0, 64, 128],
                  [255, 255, 255]]

    return colors


# load thresh from matlab result
# .mat should in the path
def load_thresh(path, cls):
    thresh = []
    for c in range(cls):
        filename = f"class_{c + 1}.mat"
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            result = sci.loadmat(file_path)
            thresh.append(result['result_cls'][1][0][0][0])
        else:
            raise ValueError(f"No {filename} in dir {path}")
    return thresh


# read flist from file
def default_flist_reader(flist):
    imlist = []
    with open(flist, "r") as rf:
        for line in rf.readlines():
            splitted = line.strip().split()
            if len(splitted) == 2:
                impath, imlabel = splitted
            elif len(splitted) == 1:
                impath, imlabel = splitted[0], None
            else:
                raise ValueError("flist line value error")
            impath = impath.strip("../")
            imseg = imlabel.replace("edge.bin", "trainIds.png")
            imlist.append(impath, imlabel, imseg)
    return imlist


# predict image
def image_predict(args, model_list, img):
    '''
    :param model_list: default(backbone, net)
    :param img: BGR - mean_value, torch.tensor, 3*H*W
    :return:
        sed_pred: torch.tensor, 1*cls*H*W, no sigmoid
        seg_pred: torch.tensor, 1*cls*H*W, no argmax
        edge_pred: torch.tensor, 1*1*H*W, no sigmoid
    '''
    # checking img.ndim
    if img.ndim == 3:
        img = img[None]
    if img.ndim != 4:
        raise ValueError("img ndim illegal")

    with torch.no_grad():
        sed_pred, seg_pred, edge_pred = run_forward(args, img, model_list[0], model_list[1])

    # custom the return
    return sed_pred[None], seg_pred, edge_pred


# predict image patches
def patch_predict(args, model_list, img, n_classes, patch_h, patch_w, step_size_y, step_size_x, pad):
    # checking img ndim
    if img.ndim == 3:
        img = img[None]
    if img.ndim != 4:
        raise ValueError(f"img ndim illegal, expect ndim in [3, 4], but got img.ndim={img.ndim}")
    n, c, h, w = img.shape
    # padding image
    assert (w - patch_w + 0.0) % step_size_x == 0, "padding image width must be divided by step_size_x"
    assert (h - patch_h + 0.0) % step_size_y == 0, "padding image height must be divided by step_size_y"
    step_num_x = int((w - patch_w + 0.0) / step_size_x) + 1
    step_num_y = int((h - patch_h + 0.0) / step_size_y) + 1

    # for overlaped inference
    img = F.pad(img, (pad, pad, pad, pad), 'constant', 0)

    # init temp result
    sed_out = torch.zeros(n, n_classes, h, w).to(args.device)
    seg_out = torch.zeros(n, n_classes, h , w).to(args.device)
    edge_out = torch.zeros(n, 1, h, w).to(args.device)
    mat_count = torch.zeros(n, 1, h, w).to(args.device)

    # do patch and merge
    for i in range(step_num_y):
        offset_y = i * step_size_y
        for j in range(step_num_x):
            offset_x = j * step_size_x
            patch_in = img[:, :, offset_y:offset_y + patch_h + 2 * pad, offset_x:offset_x + patch_w + 2 * pad]
            sed_pred, seg_pred, edge_pred = image_predict(args, model_list, patch_in)
            del patch_in
            
            sed_out[:, :, offset_y:offset_y + patch_h, offset_x:offset_x + patch_w] += \
                sed_pred[:, :, -pad: pad, -pad: pad]
            seg_out[:, :, offset_y:offset_y + patch_h, offset_x: offset_x + patch_w] += \
                seg_pred[:, :, -pad: pad, -pad: pad]
            edge_out[:, :, offset_y:offset_y + patch_h, offset_x: offset_x + patch_w] += \
                edge_pred[:, :, -pad: pad, -pad: pad]
            mat_count[:, :, offset_y:offset_y + patch_h, offset_x: offset_x + patch_w] += 1.0
            del sed_pred
            del seg_pred
            del edge_pred
    del img
    sed_out = torch.divide(sed_out, mat_count)
    seg_out = torch.divide(seg_out, mat_count)
    edge_out = torch.divide(edge_out, mat_count)

    return sed_out, seg_out, edge_out


# 长宽不被整除
def patch_predict_(args, model_list, img, n_classes, patch_h, patch_w, step_size_y, step_size_x, pad):
    '''
    :param model_list: default(backbone, net)
    :param img: BGR - mean_value, torch.Tensor, 3*H*W
    :param patch_h: int
    :param patch_w: int
    :param step_size_y: int
    :param step_size_x: int
    :param pad: int, pad is for overlaped inference
    :return: pred result
    '''
    # checking img ndim
    if img.ndim == 3:
        img = img[None]
    if img.ndim != 4:
        raise ValueError(f"img ndim illegal, expect ndim in [3, 4], but got img.ndim={img.ndim}")
    n, c, h, w = img.shape

    step_num_x = math.ceil((w - patch_w + 0.0) / step_size_x) + 1
    step_num_y = math.ceil((h - patch_h + 0.0) / step_size_y) + 1

    # for overlaped inference
    img = F.pad(img, (pad, pad, pad, pad), 'reflect')

    # init temp result
    sed_out = torch.zeros(n, n_classes, h, w).to(args.device)
    seg_out = torch.zeros(n, n_classes, h, w ).to(args.device)
    edge_out = torch.zeros(n, 1, h , w).to(args.device)
    mat_count = torch.zeros(n, 1, h, w).to(args.device)

    # do patch and merge
    for i in range(step_num_y):
        if i != step_num_y - 1:
            offset_y1 = i * step_size_y
            offset_y2_pad = offset_y1 + patch_h + 2 * pad
            offset_y2 = offset_y1 + patch_h
        else:
            offset_y1 = h - patch_h
            offset_y2_pad = h + 2 * pad
            offset_y2 = h

        for j in range(step_num_x):
            if j != step_num_x - 1:
                offset_x1 = j * step_size_x
                offset_x2_pad = offset_x1 + patch_w + 2 * pad
                offset_x2 = offset_x1 + patch_w
            else:
                offset_x1 = w - patch_w
                offset_x2_pad = w + 2 * pad
                offset_x2 = w
            patch_in = img[:, :, offset_y1:offset_y2_pad, offset_x1:offset_x2_pad]
            sed_pred, seg_pred, edge_pred = image_predict(args, model_list, patch_in)
            if sed_pred.ndim == 5:
                sed_pred = sed_pred.squeeze(0)
            del patch_in
            sed_out[:, :, offset_y1: offset_y2, offset_x1: offset_x2] += \
                sed_pred[:, :, pad: -pad, pad: -pad]
            if seg_pred is not None:
                seg_out[:, :, offset_y1: offset_y2, offset_x1: offset_x2] += \
                    seg_pred[:, :, pad: -pad, pad: -pad]
            if edge_pred is not None:
                edge_out[:, :, offset_y1: offset_y2, offset_x1: offset_x2] += \
                    edge_pred[:, :, pad: -pad, pad: -pad]
            mat_count[:, :, offset_y1: offset_y2, offset_x1: offset_x2] += 1.0
            del sed_pred
            del seg_pred
            del edge_pred
    del img
    sed_out = torch.divide(sed_out, mat_count)
    seg_out = torch.divide(seg_out, mat_count)
    edge_out = torch.divide(edge_out, mat_count)

    return sed_out, seg_out, edge_out


# save orig edge masks
def save_sed_pred_masks(sed_pred, img_info, n_classes, factor, out_folder):
    height, width = img_info["orig_size"]
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    # print(sed_pred.shape, __name__, line312)
    for i in range(n_classes):
        temp_mask = sed_pred[i, :height, :width]
        temp_mask = temp_mask * 255 * factor
        temp_mask = np.where(temp_mask > 255, 255, temp_mask)
        im = temp_mask.astype(np.uint8)
        out_path = os.path.join(out_folder, f"class_{i + 1}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path, filename), im)


# save edge masks higher than thresh
def save_sed_binary_masks(sed_pred, img_info, n_classes, thresh, out_folder):
    height, width = img_info["orig_size"]
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    for i in range(n_classes):
        temp_mask = sed_pred[i, 0:height, 0:width]
        temp_mask[temp_mask >= 255] = 0
        temp_mask = np.where(temp_mask > thresh[i], 255, 0)
        im = temp_mask.astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        out_path = os.path.join(out_folder, f"class_{i + 1}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path, filename), im)


# save visual sed result
def save_visual_sed_pred(sed_pred, n_classes, img_info, dataset, thresh, out_folder):
    height, width = img_info["orig_size"]
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    visual_img = np.zeros((height, width, 3))
    edge_sum = np.zeros((height, width))
    colors = get_visual_colors(dataset)

    for i in range(n_classes):
        temp_mask = sed_pred[i, 0:height, 0:width]
        temp_mask[temp_mask >= 255] = 0
        temp_mask = np.where(temp_mask >= thresh[i], 1, 0)
        edge_sum += temp_mask
        for c in range(3):
            visual_img[:, :, c] = np.where(temp_mask == 1, visual_img[:, :, c] + colors[i][c], visual_img[:, :, c])
    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    visual_img[idx] = visual_img[idx] / edge_sum[idx]
    visual_img[~idx] = 255

    out_file = os.path.join(out_folder, filename)
    visual_img = cv2.cvtColor(visual_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file, visual_img)


# save seg result
def save_seg_pred(seg_pred, img_info, out_folder):
    seg_result = np.argmax(seg_pred, axis=0).astype(np.uint8)
    height, width = img_info["orig_size"]
    seg_result = cv2.resize(seg_result, (width, height), interpolation=cv2.INTER_NEAREST)
    seg_result = cv2.cvtColor(seg_result, cv2.COLOR_GRAY2BGR)
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    cv2.imwrite(os.path.join(out_folder, filename), seg_result)


# save visual seg result
def save_visual_seg_pred(seg_pred, dataset, img_info, out_folder):
    seg_result = np.argmax(seg_pred, axis=0)
    colors = get_visual_colors(dataset)
    height, width = img_info["orig_size"]
    seg_result = cv2.resize(seg_result, (width, height), interpolation=cv2.INTER_NEAREST)
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    visual_seg = np.zeros((height, width, 3))
    for row in range(height):
        for col in range(width):
            visual_seg[row, col, :] = np.array(colors[int(seg_result[row, col])])

    cv2.imwrite(os.path.join(out_folder, filename), visual_seg)


# save edge pred
def save_edge_pred(edge_pred, img_info, out_folder):
    edge_pred = (edge_pred * 255).astype(np.uint8)
    height, width = img_info["orig_size"]
    edge_pred = cv2.resize(edge_pred, (width, height))
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    edge_pred = cv2.cvtColor(edge_pred, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(os.path.join(out_folder, filename), edge_pred)


def run_forward(args, images, extractor, model):
    if args.backbone == "resnet101":
        res1, res2, res3, res4, res5 = extractor(images)
    else:
        res1 = None
        res2, res3, res4, res5 = extractor(images)

    results = model(images, res1, res2, res3, res4, res5)
    if args.model_name in ["aca", "ahsi"]:
        pred_sed, pred_seg, pred_edge = results
        pred_sed = torch.sigmoid(pred_sed)
        pred_edge = torch.sigmoid(pred_edge)
    else:
        raise ValueError(f"Error: unknown model {args.model} when run_forward")
    return pred_sed, pred_seg, pred_edge


def build_model(args, feat_chans):
    if args.model_name == "ahsi":
        from models import Top_Down
        model = Top_Down(nclasses=args.n_classes, inter_channel=256, feat_chans=feat_chans)
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        raise ValueError(f"Error: unknown model_name when build model {args.model_name}")
    return model


def build_dataset(args):
    if args.dataset == "sbd":
        from utils.sbd_datasets import SBDTest
        dataset = SBDTest(args.data_root, args.file_list)
    elif args.dataset == "cityscapes":
        from utils.city_datasets import DataList
        dataset = DataList(args.data_root, args.file_list, args.n_classes)
    elif args.dataset == "ade20k":
        from utils.ade20k.ade20k_datasets import DataList
        dataset = DataList(args.data_root, args.file_list, args.n_classes)
    else:
        raise ValueError(f"unknown dataset, {args.dataset}")

    return dataset

def main(args):
    print("****")
    print(args)
    print("****")

    dataset_ = build_dataset(args)
    file_num = len(dataset_)

    model_list = nn.ModuleList()

    backbone, feat_chans = build_extractor(args)  # args.backbone
    net = build_model(args, feat_chans)  # args.model_name

    ckpt = torch.load(args.ckpt)
    backbone.load_state_dict(ckpt["extractor"], strict=True)
    net.load_state_dict(ckpt["model"], strict=True)

    backbone.eval()
    net.eval()

    model_list.append(backbone)
    model_list.append(net)

    if args.thresh is not None:
        thresh = load_thresh(args.thresh, args.n_classes)
    else:
        thresh = [0.30] * args.n_classes

    for i in range(len(thresh)):
        thresh[i] = thresh[i] if thresh[i] > 0.2 else 0.2

    for idx in tqdm.tqdm(range(file_num)):
        img, gt, seg, edge, img_info = dataset_[idx]

        if not args.use_patch_pred:
            sed_pred, seg_pred, edge_pred = image_predict(args, model_list, img.to(args.device))
        else:
            sed_pred, seg_pred, edge_pred = patch_predict_(args, model_list, img.to(args.device), n_classes=args.n_classes,
                                                          patch_h=args.patch_h, patch_w=args.patch_w, step_size_x=args.step_x, step_size_y=args.step_y,
                                                          pad=args.pad)
            sed_pred = F.interpolate(sed_pred, size=img_info['orig_size'], mode='bilinear')

        sed_pred = sed_pred.squeeze(0)
        sed_pred = torch.squeeze(sed_pred, 0).cpu().numpy()
        out_folder = os.path.join(args.out_dir, "classes")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        save_sed_pred_masks(sed_pred, img_info, args.n_classes, args.factor, out_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="project root")
    parser.add_argument("--model-name", type=str, default="aca", choices=['ahsi', 'aca', 'aca_mask_cls', 'only_mask_cls', 'only_mask_cls2', 'upernet'])
    parser.add_argument("--rank", type=int, default=256, help="Query number in ACA")
    parser.add_argument("--n_queries", type=int, default=40, help="query number for aca_mask_cls")
    parser.add_argument("--num_aux_layers", type=int, default=6, help="deep supervise layer in aca_mask_cls")
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--vit-name", type=str, default='dinov2_large')
    parser.add_argument("--spm-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "sbd", "ade20k"],
                        help="specify dataset name")
    parser.add_argument("--data_root", type=str, default="dataset_name/data_proc", help="gtFine and leftImg8bit in dir")
    parser.add_argument("--file_list", type=str, default="dataset_name/data_proc/val.txt",
                        help="Line format: image_path edge_bin_path")
    parser.add_argument("--n_classes", type=int, help="number of classes")
    parser.add_argument("--ckpt", type=str, help="ckpt file path")
    parser.add_argument("--factor", type=float, default=1.0, help="factor on results")
    parser.add_argument("--thresh", type=str, default=None, help=".mat result files in the dir")
    parser.add_argument("--out_dir", type=str, default=None, required=True, help="root dir of outputs")
    parser.add_argument("--use_patch_pred", action='store_true', default=False)
    parser.add_argument("--patch_h", type=int, default=640)
    parser.add_argument("--patch_w", type=int, default=640)
    parser.add_argument("--step_x", type=int, default=352)
    parser.add_argument("--step_y", type=int, default=384)
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vit-top-feats", action='store_true', default=False)
    parser.add_argument("--freeze-vit", action='store_true', default=True)
    parser.add_argument("--eval", action='store_true', default=True)
    parser.add_argument("--mask-attn-t", type=float, default=0.2)

    args = parser.parse_args()

    if args.root is not None:
        os.chdir(args.root)
        sys.path.append(args.root)

    main(args)
