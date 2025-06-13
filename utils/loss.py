# coding:utf-8
import torch
import numpy as np
from .criterion import SetCriterion
from .matcher import HungarianMatcher

class criterion_all:
    def __init__(self, args, losses_train, losses_val, set_criterion_losses):
        self.state = "train"
        self.losses_train = losses_train
        self.losses_val = losses_val

        self.losses = losses_train.copy()
        for loss in self.losses_val:
            if loss not in self.losses:
                self.losses.append(loss)

        # 可选 weight：loss_ce, loss_mask, loss_dice
        self.weight_dict = {"loss_ce": args.cls_weight, "loss_sed": args.sed_weight,
                            "loss_edge_train": args.loss_edge,
                            "loss_edge_val": args.loss_edge,
                            "loss_dice_sed": args.dice_sed_weight,
                            "loss_seg_train": args.loss_seg,
                            "loss_seg_val": args.loss_seg,
                            "loss_sed_sbd_train2": 1,
                            "loss_sed_sbd_val2": 1
                            }

        # 设置深监督的损失
        aux_weight_dict = {}
        aux_keys = ["loss_ce", "loss_sed", "loss_dice_sed"]
        for i in range(args.num_aux_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items() if k in aux_keys})
        self.weight_dict.update(aux_weight_dict)
        print("self.losses: ", self.losses)
        self.loss_map = {}

        if "loss_seg_train" in self.losses or "loss_seg_val" in self.losses:
            criterion_seg = torch.nn.DataParallel(Seg_CrossEntropy()).cuda()
            self.loss_map["loss_seg_train"] = criterion_seg
            self.loss_map["loss_seg_val"] = criterion_seg

        if "loss_edge_train" in self.losses:
            weights_edge = torch.Tensor(np.zeros((args.batch_size, 1, args.crop_size, args.crop_size))).cuda()
            criterion_edge = torch.nn.DataParallel(Edge_bce(weights_edge)).cuda()
            self.loss_map["loss_edge_train"] = criterion_edge

        if "loss_edge_val" in self.losses:
            weights_edge_val = torch.Tensor(np.zeros((1, 1, args.crop_size, args.crop_size))).cuda()
            criterion_edge_val = torch.nn.DataParallel(Edge_bce(weights_edge_val)).cuda()
            self.loss_map["loss_edge_val"] = criterion_edge_val

        if "loss_sed_sbd_train" in self.losses:
            weights = torch.Tensor(np.zeros((args.batch_size, args.nclasses, args.crop_size, args.crop_size))).cuda()
            criterion = torch.nn.DataParallel(SBDAutoCrossEntropyLoss(weights)).cuda()
            self.loss_map["loss_sed_sbd_train"] = criterion

        if "loss_sed_sbd_val" in self.losses:
            weights_val = torch.Tensor(np.zeros((1, args.nclasses, args.crop_size, args.crop_size))).cuda()
            criterion_val = torch.nn.DataParallel(SBDAutoCrossEntropyLoss(weights_val)).cuda()
            self.loss_map["loss_sed_sbd_val"] = criterion_val

        if "loss_sed_city_train" in self.losses:
            weights = torch.Tensor(np.zeros([args.batch_size, args.nclasses, args.crop_size, args.crop_size])).cuda()
            criterion = torch.nn.DataParallel(CityAutoCrossEntropyLoss(weights)).cuda()
            self.loss_map["loss_sed_city_train"] = criterion

        if "loss_sed_city_val" in self.losses:
            weights = torch.Tensor(np.zeros([1, args.nclasses, args.crop_size, args.crop_size])).cuda()
            criterion = torch.nn.DataParallel(CityAutoCrossEntropyLoss(weights)).cuda()
            self.loss_map['loss_sed_city_val'] = criterion

        if "loss_sed_ade20k_train" in self.losses:
            weights = torch.Tensor(np.zeros([args.batch_size, args.nclasses, args.crop_size, args.crop_size])).cuda()
            criterion = torch.nn.DataParallel(SBDAutoCrossEntropyLoss(weights)).cuda()
            self.loss_map["loss_sed_city_train"] = criterion

        if "loss_sed_ade20k_val" in self.losses:
            weights = torch.Tensor(np.zeros([1, args.nclasses, args.crop_size, args.crop_size])).cuda()
            criterion = torch.nn.DataParallel(SBDAutoCrossEntropyLoss(weights)).cuda()
            self.loss_map['loss_sed_city_val'] = criterion

        if "loss_set_criterion" in self.losses:
            mask_matcher = HungarianMatcher(
                cost_class=self.weight_dict["loss_ce"],
                cost_sed=self.weight_dict["loss_sed"],
                cost_dice_sed=self.weight_dict['loss_dice_sed']
            )
            loss_set_criterion = SetCriterion(
                args.num_iters,
                num_classes=args.nclasses,
                matcher=mask_matcher,
                weight_dict=self.weight_dict,
                no_obj_ce=args.no_object_weight,
                no_obj_bce=args.no_object_bce_weight,
                losses=set_criterion_losses,
            )
            self.loss_map["loss_set_criterion"] = loss_set_criterion

    def train(self):
        self.state = "train"

    def training(self):
        return self.state == "train"

    def eval(self):
        self.state = "eval"

    def get_loss(self, loss, outputs, targets, **kwargs):
        assert loss in self.loss_map, f"no loss {loss} in init losses {str(self.loss_map.keys())}"
        return self.loss_map[loss](outputs, targets, **kwargs)

    def __call__(self, outputs, targets, **kwargs):
        losses = {}
        # 根据当前的状态求损失
        if self.training():
            temp_loss_keys = self.losses_train
        else:
            temp_loss_keys = self.losses_val
        for loss in temp_loss_keys:
            if loss in ["loss_sed_city_train_0", "loss_sed_city_val_0"]:
                result = self.get_loss(loss, outputs['aux_outputs'][0], targets, **kwargs)
            elif loss in ["loss_sed_city_val_1", "loss_sed_city_train_1"]:
                result = self.get_loss(loss, outputs['aux_outputs'][1], targets, **kwargs)
            else:
                result = self.get_loss(loss, outputs, targets, **kwargs)
            # result = self.get_loss(loss, outputs, targets, **kwargs)
            if not isinstance(result, dict):
                result = {loss: result}
            losses.update(result)
        return losses


# ===================================SBD=============================================
# 带权重
class SBDWeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights):
        super(SBDWeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        n, c, h, w = outputs.size()
        for i in xrange(n):
            t = targets[i, :, :, :]
            edge_map = t.max(0)[0]  # 0才是结果
            pos = (edge_map == 1).sum()
            neg = (edge_map == 0).sum()
            valid = neg + pos
            self.weights[i, t == 1] = neg * 1. / valid
            self.weights[i, t == 0] = pos * 1. / valid
            self.weights[i, t == 255] = 0.
        inputs = torch.sigmoid(outputs)
        loss = torch.nn.BCELoss(self.weights, reduction='sum')(inputs, targets)
        return loss


# 自动权重
class SBDAutoCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights, pred_key='pred_sed', tgt_key='gt_sed'):
        super(SBDAutoCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.pred_key = pred_key
        self.tgt_key = tgt_key
        self.weight_prior = torch.as_tensor([1.661302570403549, 1.8258168845012133, 1.7380136133185058, 2.21283413032427,
                                        2.365753439726141, 2.0906042392547617, 1.3884012075693861, 1.1908220653601478,
                                        1.2053603146627756, 2.508736280602481, 2.056573575856723, 1.1023722999575336,
                                        1.7465868518029228, 1.8326368020584554, 0.3851410026643433, 2.352477618281501,
                                        2.3337779067889555, 1.8031714731538628, 1.9806775824566383, 2.4130427386526883])

    def forward(self, outputs, targets, cur_iter, max_iter):
        output = outputs[self.pred_key] if self.pred_key is not None else outputs
        target = targets[self.tgt_key] if self.tgt_key is not None else targets

        n, c, h, w = output.size()
        w_pos = 0.98 - (0.98 - 0.5) * min(cur_iter * 1.25 / max_iter, 1)
        w_neg = 0.02 + (0.5 - 0.02) * min(cur_iter * 1.25 / max_iter, 1)
        for i in range(n):
            self.weights[i, target[i] == 1] = w_pos * self.weight_prior[i]
            self.weights[i, target[i] == 0] = w_neg * self.weight_prior[i]
            self.weights[i, target[i] == 255] = 0.
        output = torch.sigmoid(output)
        target = target.float()

        mask = target == 255
        # output[mask] = 0
        target[mask] = 0
        # print("Output min:", output.min().item())
        # print("Output max:", output.max().item())
        # print("Target min:", target.min().item())
        # print("Target max:", target.max().item())
        loss = torch.nn.BCELoss(self.weights, reduction='sum')(output, target)
        return loss


class AutoCrossEntropyLossWeight(torch.nn.Module):
    def __init__(self, args, pred_key='pred_sed', tgt_key='gt_sed', val=False):
        super(AutoCrossEntropyLossWeight, self).__init__()
        self.weights = torch.Tensor(
            np.zeros((1 if val else args.batch_size, args.nclasses, args.crop_size, args.crop_size))
        ).to(args.device)
        self.pred_key = pred_key
        self.tgt_key = tgt_key
        if args.dataset == "sbd":
            self.weight_prior = torch.as_tensor([1.661302570403549, 1.8258168845012133, 1.7380136133185058, 2.21283413032427,
                                        2.365753439726141, 2.0906042392547617, 1.3884012075693861, 1.1908220653601478,
                                        1.2053603146627756, 2.508736280602481, 2.056573575856723, 1.1023722999575336,
                                        1.7465868518029228, 1.8326368020584554, 0.3851410026643433, 2.352477618281501,
                                        2.3337779067889555, 1.8031714731538628, 1.9806775824566383, 2.4130427386526883])
        elif args.dataset == "cityscapes":
            self.weight_prior = torch.as_tensor(
                [0.6687723048974245, 0.9428777678592765, 0.48523797980417893, 3.14681474430984, 2.736030954214118,
                 0.7842433181425774, 3.2445364063017808, 2.432837362686337, 0.5951157725549547, 2.2576593860563157,
                 1.6536053475868233, 1.6935741342664692, 4.012557128464438, 1.0576898152026029, 4.482716597937749,
                 4.513237092242108, 4.605448686899701, 4.481463972345664, 3.000203176437492]
            )

    def forward(self, outputs, targets, cur_iter, max_iter):
        output = outputs[self.pred_key] if self.pred_key is not None else outputs
        target = targets[self.tgt_key] if self.tgt_key is not None else targets
        if isinstance(output, tuple):
            output = output[0]
        # print(output)
        # raise ValueError
        n, c, h, w = output.size()
        w_pos = 0.98 - (0.98 - 0.5) * min(cur_iter * 1.25 / max_iter, 1)
        w_neg = 0.02 + (0.5 - 0.02) * min(cur_iter * 1.25 / max_iter, 1)
        for i in range(n):
            self.weights[i, target[i] == 1] = w_pos * self.weight_prior[i]
            self.weights[i, target[i] == 0] = w_neg * self.weight_prior[i]
            self.weights[i, target[i] == 255] = 0.
        output = torch.sigmoid(output)
        target = target.float()

        mask = target == 255
        # output[mask] = 0
        target[mask] = 0
        loss = torch.nn.BCELoss(self.weights, reduction='sum')(output, target)
        return loss


class Seg_CrossEntropy(torch.nn.Module):
    def __init__(self, pred_key='pred_seg', tgt_key='gt_seg'):
        super(Seg_CrossEntropy, self).__init__()
        self.pred_key = pred_key
        self.tgt_key = tgt_key
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction='sum',
            ignore_index=255
        )

    def forward(self, outputs, targets, **kwargs):
        if self.pred_key is not None:
            score = outputs[self.pred_key]
        else:
            score = outputs

        if self.tgt_key is not None:
            target = targets[self.tgt_key]
        else:
            target = targets

        target = torch.as_tensor(target, dtype=torch.long)
        target = target.cuda()
        loss = self.criterion(score, target)
        return loss


class Seg_imagebased_crossentropy(torch.nn.Module):
    def __init__(self):
        super(Seg_imagebased_crossentropy, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight=None, reduction='sum', ignore_index=255)

    def calculate_weight(self, target, classes):
        bins = torch.histc(target, bins=classes, min=0, max=classes - 1)

        hist_norm = bins.float() / bins.sum()
        hist = ((bins != 0).float() * (1.0 - hist_norm)) + 1.0

        return hist

    def forward(self, score, target):
        n, c, h, w = score.size()

        target = torch.as_tensor(target, dtype=torch.long)
        target = target.cuda()

        loss = 0.0
        for i in range(n):
            weights = self.calculate_weight(target[i], c)
            self.nll_loss.weight = weights
            loss += self.nll_loss(torch.nn.functional.log_softmax(score[i].unsqueeze(0), dim=1), target[i].unsqueeze(0))

        return loss


class Edge_bce(torch.nn.Module):
    def __init__(self, weights, pred_key='pred_edge', tgt_key='gt_edge'):
        super(Edge_bce, self).__init__()
        self.weights = weights
        self.pred_key=pred_key
        self.tgt_key=tgt_key

    def forward(self, outputs, targets, **kwargs):
        if self.pred_key is not None:
            output = outputs[self.pred_key]
        else:
            output = outputs
        if self.tgt_key is not None:
            target = targets["gt_edge"]
        else:
            target = targets

        n, c, h, w = output.size()
        for i in range(n):
            t = target[i, :, :, :]
            pos = (t == 1).sum()
            neg = (t == 0).sum()
            valid = neg + pos
            self.weights[i, t == 1] = neg * 1. / valid
            self.weights[i, t == 0] = pos * 1. / valid
            self.weights[i, t == 255] = 0.
        input = torch.sigmoid(output)
        mask = target == 255
        # input[mask] = 0
        target[mask] = 0
        loss = torch.nn.BCELoss(self.weights, reduction='sum')(input, target)
        return loss


class ConsistencyConstraintLossGT(torch.nn.Module):
    def __init__(self):
        super(ConsistencyConstraintLossGT, self).__init__()

    def forward(self, outputs, targets):
        output_dict = outputs['pred_const_dict']
        edge_embed = output_dict['edge_embed']
        pixel_feat = output_dict['pixel_feat']  # b C h w, C=256
        targets = targets['gt_sed']  # b c h w, c=classes
        targets = targets.clone()
        targets = torch.where(targets == 1, 1., 0.)
        targets = torch.nn.functional.interpolate(targets, size=pixel_feat.shape[-2:], mode='nearest', align_corners=True)


class Weight_SBDAutoCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights):
        super(Weight_SBDAutoCrossEntropyLoss, self).__init__()
        self.weights = weights
        n, c, h, w = weights.size()
        weight_list = []
        for i in range(n):
            weight_list.append(
                [1.661302570403549, 1.8258168845012133, 1.7380136133185058, 2.21283413032427,
                 2.365753439726141, 2.0906042392547617, 1.3884012075693861, 1.1908220653601478,
                 1.2053603146627756, 2.508736280602481, 2.056573575856723, 1.1023722999575336,
                 1.7465868518029228, 1.8326368020584554, 0.3851410026643433, 2.352477618281501,
                 2.3337779067889555, 1.8031714731538628, 1.9806775824566383, 2.4130427386526883])
        weights1 = torch.Tensor(np.array(weight_list)).cuda()
        self.weights1 = weights1.view(n, c, 1, 1)

    def forward(self, outputs, targets, cur_iter, max_iter):
        n, c, h, w = outputs.size()
        w_pos = 0.98 - (0.98 - 0.5) * min(cur_iter * 1.25 / max_iter, 1)
        w_neg = 0.02 + (0.5 - 0.02) * min(cur_iter * 1.25 / max_iter, 1)
        for i in xrange(n):
            self.weights[i, targets[i] == 1] = w_pos
            self.weights[i, targets[i] == 0] = w_neg
            self.weights[i, targets[i] == 255] = 0.
        self.weights = self.weights * self.weights1
        inputs = torch.sigmoid(outputs)
        loss = torch.nn.BCELoss(self.weights, reduction='sum')(inputs, targets)
        return loss


# ===================================City=============================================
class CityWeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights1):
        super(CityWeightedCrossEntropyLoss, self).__init__()
        self.weights1 = weights1
        n, c, h, w = weights1.size()
        weight_list = []
        for i in range(n):
            weight_list.append(
                [0.6687723048974245, 0.9428777678592765, 0.48523797980417893, 3.14681474430984, 2.736030954214118,
                 0.7842433181425774, 3.2445364063017808, 2.432837362686337, 0.5951157725549547, 2.2576593860563157,
                 1.6536053475868233, 1.6935741342664692, 4.012557128464438, 1.0576898152026029, 4.482716597937749,
                 4.513237092242108, 4.605448686899701, 4.481463972345664, 3.000203176437492])
        weights2 = torch.Tensor(np.array(weight_list)).cuda()
        self.weights2 = weights2.view(n, c, 1, 1)

    def forward(self, outputs, targets):
        n, c, h, w = outputs.size()
        for i in range(n):
            t = targets[i, :, :, :]
            edge_map = t.max(0)[0]  # 0才是结果
            pos = (edge_map == 1).sum()
            neg = (edge_map == 0).sum()
            valid = neg + pos
            self.weights1[i, t == 1] = neg * 1. / valid
            self.weights1[i, t == 0] = pos * 1. / valid
            self.weights1[i, t == 255] = 0.
        self.weights1 = self.weights1 * self.weights2
        inputs = torch.sigmoid(outputs)
        loss = torch.nn.BCELoss(self.weights1, reduction='sum')(inputs, targets)
        return loss


# 自动权重
class CityAutoCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights1, pred_key='pred_sed', tgt_key='gt_sed'):
        super(CityAutoCrossEntropyLoss, self).__init__()
        self.weights1 = weights1
        self.pred_key = pred_key
        self.tgt_key = tgt_key
        n, c, h, w = weights1.size()
        weight_list = []
        for i in range(n):
            weight_list.append(
                [0.6687723048974245, 0.9428777678592765, 0.48523797980417893, 3.14681474430984, 2.736030954214118,
                 0.7842433181425774, 3.2445364063017808, 2.432837362686337, 0.5951157725549547, 2.2576593860563157,
                 1.6536053475868233, 1.6935741342664692, 4.012557128464438, 1.0576898152026029, 4.482716597937749,
                 4.513237092242108, 4.605448686899701, 4.481463972345664, 3.000203176437492])
        weights2 = torch.Tensor(np.array(weight_list)).cuda()
        self.weights2 = weights2.view(n, c, 1, 1)

    def forward(self, outputs, targets, cur_iter, max_iter):
        outputs = outputs[self.pred_key] if self.pred_key is not None else outputs
        targets = targets[self.tgt_key] if self.tgt_key is not None else targets

        n, c, h, w = outputs.size()
        w_pos = 0.98 - (0.98 - 0.5) * min(cur_iter * 1.25 / max_iter, 1)
        w_neg = 0.02 + (0.5 - 0.02) * min(cur_iter * 1.25 / max_iter, 1)
        for i in range(n):
            self.weights1[i, targets[i] == 1] = w_pos
            self.weights1[i, targets[i] == 0] = w_neg
            self.weights1[i, targets[i] == 255] = 0.
        self.weights1 = self.weights1 * self.weights2
        inputs = torch.sigmoid(outputs)
        loss = torch.nn.BCELoss(self.weights1, reduction='sum')(inputs, targets)
        return loss


class Seg_CrossEntropy_City(torch.nn.Module):
    def __init__(self):
        super(Seg_CrossEntropy_City, self).__init__()

        self.weight = torch.from_numpy(np.array([
            0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969,
            0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843,
            1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
            1.0507
        ])).float()
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=self.weight,
            reduction='sum',
            ignore_index=255
        )

    def forward(self, score, target):
        target = torch.as_tensor(target, dtype=torch.long)
        target = target.cuda()

        loss = self.criterion(score, target)

        return loss


if __name__ == "__main__":
    # loss
    batch_size = 0
    NUM_CLASSES = 0
    crop_size = 0

    weights = torch.Tensor(np.zeros((batch_size, NUM_CLASSES, crop_size, crop_size))).cuda()
    criterion = torch.nn.DataParallel(SBDAutoCrossEntropyLoss(weights)).cuda()

    criterion_seg = torch.nn.DataParallel(Seg_CrossEntropy()).cuda()

    # weights_edge = torch.Tensor(np.zeros((args.batch_size, 1, args.crop_size/2, args.crop_size/2))).cuda()
    weights_edge = torch.Tensor(np.zeros((batch_size, 1, crop_size, crop_size))).cuda()
    criterion_edge = torch.nn.DataParallel(Edge_bce(weights_edge)).cuda()
