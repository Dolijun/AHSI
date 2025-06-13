# coding:utf-8
from torch.utils.data import Dataset
import cv2
import random
import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
from utils.dataset_utils import get_output_shape
from utils.color_augmentation import ColorAugSSDTransform


class DataList(Dataset):
    def __init__(self, root, flist, n_classes):
        self.root = root
        self.imlist = self.default_flist_reader(flist)
        self.n_classes = n_classes
        self.mean_value = (104.008, 116.669, 122.675)  # BGR
        self.short_edge_length = 1024
        self.max_length = 2560

    # 1.读取文件索引列表 .txt格式(img,label)
    def default_flist_reader(self, flist):
        """
        flist format: impath imseg label\nimpath imseg label\n ...
        """
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                splitted = line.strip().split()
                if len(splitted) == 3:
                    impath, imseg, imlabel = splitted
                else:
                    raise ValueError('weird length ?')
                imlist.append((impath, imlabel, imseg))
        return imlist

    # 2.把位真值转换成通道真值
    # def binary_file_to_channel_masks(self, bin_file, h, w, channels, ignore_pixel_id_map=(31, 255)):
    #     array = np.fromfile(bin_file, dtype=np.uint32)
    #     array = array.reshape(h, w)
    #     arr_chn = np.zeros((channels, h, w), dtype=np.float32)
    #     ignore_mask = array & (1 << ignore_pixel_id_map[0]) > 0
    #     for c in range(channels):
    #         mask = array & (1 << c) > 0
    #         arr_chn[c, :, :] = mask
    #         arr_chn[c, :, :][ignore_mask] = ignore_pixel_id_map[1]
    #     return arr_chn.transpose((1, 2, 0))

    # def sedge_int32_to_channel_masks(self, sedge_int32, channels, ignore_pixel_id_map=(-1, 31, 255)):
    #     sedge_int32_array = np.load(sedge_int32)
    #     h, w = sedge_int32_array.shape[-2:]
    #     sedge_chan = np.zeros((channels, h, w), dtype=np.float32)
    #     ignore_mask = sedge_int32_array[ignore_pixel_id_map[0]] & (1 << ignore_pixel_id_map[1]) > 0
    #     for c in range(channels):
    #         split = c // 32
    #         chan = c % 32
    #         mask = sedge_int32_array[split] & (1 << chan) > 0
    #         sedge_chan[c, :, :] = mask
    #         # sedge_chan[c, :, :][ignore_mask] = ignore_pixel_id_map[-1]
    #     return sedge_chan.transpose((1, 2, 0))

    def sparse_dict_to_channel_masks(self, sparse_npy, n_classes=150, ignore_id=255):
        sparse_dict = np.load(sparse_npy, allow_pickle=True).item()
        ignore_mask = sparse_dict[-1]
        del sparse_dict[-1]
        h, w = ignore_mask.shape
        sed_chan = np.zeros((n_classes, h, w), dtype=np.float32)
        for c, mask in sparse_dict.items():
            sed_chan[c, :, :] = mask
            sed_chan[c, :, :][ignore_mask] = ignore_id
        return sed_chan.transpose((1, 2, 0))

    # 3.图片预处理
    def my_transform(self, in_, gt_mask, seg):
        h, w = in_.shape[:2]
        short_edge_length = self.short_edge_length
        tgt_size = get_output_shape(h, w, short_edge_length, self.max_length)
        in_ = cv2.resize(in_, dsize=tgt_size, interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask, dsize=tgt_size, interpolation=cv2.INTER_NEAREST)
        seg = cv2.resize(seg, dsize=tgt_size, interpolation=cv2.INTER_NEAREST)
        return in_, gt_mask, seg

    # 4.裁剪
    def auto_crop(self, crop_size, input, target, seg, is_fix=False):
        img_width, img_height = input.shape[1], input.shape[0]
        pad_height = max(crop_size - img_height, 0)
        pad_width = max(crop_size - img_width, 0)
        if pad_height > 0 or pad_width > 0:
            input = cv2.copyMakeBorder(input, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)
            target = cv2.copyMakeBorder(target, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[255] * 4)
            seg = cv2.copyMakeBorder(seg, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=255)
        width, height = input.shape[1], input.shape[0]
        transX = random.randint(0, width - crop_size)
        transY = random.randint(0, height - crop_size)
        if is_fix:
            transX = 0
            transY = 0
        input = input[transY:transY + crop_size, transX:transX + crop_size, :]
        target = target[transY:transY + crop_size, transX:transX + crop_size, :]
        seg = seg[transY:transY + crop_size, transX:transX + crop_size]
        return np.array(input), np.array(target), np.array(seg)

    def __getitem__(self, index):
        impath, gtpath, imseg = (
            os.path.join(self.root, *self.imlist[index][0].split('/')),
            os.path.join(self.root, *self.imlist[index][1].split('/')),
            os.path.join(self.root, *self.imlist[index][2].split('/'))
        )
        image = cv2.imread(impath).astype(np.float32)
        width, height = image.shape[1], image.shape[0]
        # gt = self.sedge_int32_to_channel_masks(gtpath, self.n_classes)
        gt = self.sparse_dict_to_channel_masks(gtpath, n_classes=self.n_classes)
        seg_mask = cv2.imread(imseg, 0)  # 0-18 class 255 ignore
        # -1
        seg_mask = seg_mask.astype(np.uint8)
        # print(np.unique(seg_mask))
        image, gt, seg = self.my_transform(image, gt, seg_mask)
        image -= np.array(self.mean_value)
        image = image.transpose((2, 0, 1))  # HxWx3 -> 3xHxW
        gt = gt.transpose((2, 0, 1))  # 19,H,W
        edge = np.max(gt, axis=0)  # 1 h w
        image = torch.from_numpy(image)
        gt = torch.from_numpy(gt)
        seg = torch.from_numpy(seg)
        edge = torch.from_numpy(edge)
        edge = edge.unsqueeze(0)
        image_info = {'impath': impath, 'gtpath': gtpath, 'orig_size': (height, width)}
        return image, gt, seg, edge, image_info

    def __len__(self):
        return len(self.imlist)


class ADE20kTrain(DataList):
    def __init__(self, root, flist, crop_size=512):
        super(ADE20kTrain, self).__init__(root, flist, n_classes=150)
        self.crop_size = crop_size
        # self.scale = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
        self.short_edge_length = (320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 1024, 1088, 1152, 1216, 1280)
        self.max_length = 2560
        self.color_aug = ColorAugSSDTransform()

    def my_transform(self, input, target, seg):
        # Random hflip
        if random.randrange(2):
            input = np.fliplr(input)
            target = np.fliplr(target)
            seg = np.fliplr(seg)

        # resize factor
        # scale = random.choice(self.scale)
        # input = cv2.resize(input, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # target = cv2.resize(target, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        # seg = cv2.resize(seg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # resize shortest edge
        h, w = input.shape[:2]
        short_edge_length = random.choice(self.short_edge_length)
        tgt_size = get_output_shape(h, w, short_edge_length, self.max_length)
        input = cv2.resize(input, dsize=tgt_size, interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, dsize=tgt_size, interpolation=cv2.INTER_NEAREST)
        seg = cv2.resize(seg, dsize=tgt_size, interpolation=cv2.INTER_NEAREST)

        # Random crop
        input, target, seg = self.auto_crop(self.crop_size, input, target, seg)

        # 颜色增强
        # input = self.color_aug.apply_image(input)

        input = input.astype(np.float32)
        return input, target, seg


class ADE20kVal(DataList):
    def __init__(self, root, flist, crop_size=512):
        super(ADE20kVal, self).__init__(root, flist, n_classes=150)
        self.crop_size = crop_size
        self.short_edge_length = 640
        self.max_length = 2560

    def my_transform(self, input, target, seg):
        # resize
        # h, w = input.shape[-2:]
        # short_edge_length = self.short_edge_length
        # tgt_size = get_output_shape(h, w, short_edge_length, self.max_length)
        # input = cv2.resize(input, tgt_size, interpolation=cv2.INTER_LINEAR)
        # target = cv2.resize(target, tgt_size, interpolation=cv2.INTER_NEAREST)
        # seg = cv2.resize(seg, tgt_size, interpolation=cv2.INTER_NEAREST)

        input, target, seg = self.auto_crop(self.crop_size, input, target, seg, is_fix=True)
        return input, target, seg

    def train_data(self, index):
        input, target, seg, edge, _ = self.__getitem__(index)
        input, target, seg, edge = input.unsqueeze(dim=0), target.unsqueeze(dim=0), seg.unsqueeze(
            dim=0), edge.unsqueeze(dim=0)  # 1x3xHxW
        return input, target, seg, edge


class ADE20kTest(DataList):
    def __init__(self, root, flist):
        super(ADE20kTest, self).__init__(root, flist, n_classes=150)
        self.short_edge_length = 640
        self.max_length = 2560

    def my_transform(self, input, target, seg):
        # resize
        # h, w = input.shape[-2:]
        # short_edge_length = self.short_edge_length
        # tgt_size = get_output_shape(h, w, short_edge_length, self.max_length)
        # input = cv2.resize(input, tgt_size, interpolation=cv2.INTER_LINEAR)
        # target = cv2.resize(target, tgt_size, interpolation=cv2.INTER_NEAREST)
        # seg = cv2.resize(seg, tgt_size, interpolation=cv2.INTER_NEAREST)

        # input, target, seg = self.auto_crop(self.crop_size, input, target, seg, is_fix=True)
        return input, target, seg

    def train_data(self, index):
        input, target, seg, edge, img_info = self.__getitem__(index)
        input, target, seg, edge = input.unsqueeze(dim=0), target.unsqueeze(dim=0), seg.unsqueeze(
            dim=0), edge.unsqueeze(dim=0)  # 1x3xHxW
        return input, target, seg, edge, img_info

if __name__ == '__main__':
    root = "/media/ilab/a28604eb-570f-4fb9-97fe-93fd8a7b0922/datasets/ADE20K/ADE20K_SED"
    flist = "/media/ilab/a28604eb-570f-4fb9-97fe-93fd8a7b0922/datasets/ADE20K/ADE20K_SED/training.txt"

    train_dataset = ADE20kTrain(root, flist)
    # valdataset = ADE20kVal(root, flist)

    img = train_dataset[0][0]
    cv2.imwrite(
        "/home/ilab/dolijun/Dolijun/Yan2/sed/sed_mask_classification/save_disk/ade20k/aca_mask_cls/crop512_b2/image6.jpg",
        img.cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
