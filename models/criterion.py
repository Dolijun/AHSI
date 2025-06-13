# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
from utils.misc import nested_tensor_from_tensor_list, is_dist_avail_and_initialized
import einops

def cal_similarity(key_embeds,
                   ref_embeds,
                   method='dot_product',
                   temperature=-1):
    assert method in ['dot_product', 'cosine']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="sum")

    return loss


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


#

def batch_sed_focal_loss(inputs, targets, ignore_mask, target_label, num_masks: float, max_iters: int, iters: int,
                         no_object_loss: float = 0.1, alpha: float = 0.25, gamma: int = 2):
    sbd_weight_prior = torch.as_tensor([1.661302570403549, 1.8258168845012133, 1.7380136133185058, 2.21283413032427,
                                        2.365753439726141, 2.0906042392547617, 1.3884012075693861, 1.1908220653601478,
                                        1.2053603146627756, 2.508736280602481, 2.056573575856723, 1.1023722999575336,
                                        1.7465868518029228, 1.8326368020584554, 0.3851410026643433, 2.352477618281501,
                                        2.3337779067889555, 1.8031714731538628, 1.9806775824566383, 2.4130427386526883,
                                        no_object_loss], device=targets.device)
    weight_prior = sbd_weight_prior
    print("********************** loss weight prior *************************")
    print(weight_prior)
    n, d = targets.shape
    w_pos = 0.98 - (0.98 - 0.5) * min(iters * 1.25 / max_iters, 1)
    w_neg = 0.02 + (0.5 - 0.02) * min(iters * 1.25 / max_iters, 1)

    w_pos = torch.tensor(w_pos, device=targets.device)
    w_neg = torch.tensor(w_neg, device=targets.device)
    weight = torch.zeros_like(targets).to(targets.device)
    predictions = torch.sigmoid(inputs)
    pt = torch.where(targets == 1, predictions, 1 - predictions)
    alpha_t = alpha * (1 - pt) ** gamma
    alpha_t[ignore_mask != 0] = 0
    alpha_t = alpha_t.to(targets.device).requires_grad_(True)
    for i in range(n):
        weight[i, targets[i] == 1] = w_pos * weight_prior[target_label[i]]
        weight[i, targets[i] == 0] = w_neg * weight_prior[target_label[i]]
        weight[i, ignore_mask[i] != 0] = 0

    loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction='none')
    loss = (loss * alpha_t).sum()

    del weight
    del alpha_t
    del predictions

    return loss


batch_sed_focal_loss_jit = batch_sed_focal_loss


def batch_edge_bce_loss(inputs, targets, ignore_mask):
    n, b = inputs.shape
    weight = torch.zeros_like(targets)
    for i in range(n):
        t = targets[i, :]
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = pos + neg
        weight[i, t == 1] = neg * 1.0 / valid
        weight[i, t == 0] = pos * 1.0 / valid
        weight[i, ignore_mask[i] != 0] = 0

    loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction='sum')
    del weight
    return loss


batch_edge_bce_loss_jit = torch.jit.script(
    batch_edge_bce_loss
)  # type: torch.jit.ScriptModule


class SetCriterion(nn.Module):

    def __init__(self, max_iters, num_classes, matcher, weight_dict, no_obj_ce, no_obj_bce, losses):

        super().__init__()
        self.max_iters = max_iters
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = no_obj_bce
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = no_obj_ce
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):

        assert "pred_logits" in outputs
        targets = targets['instances']
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  self.empty_weight.to(target_classes.device))
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_edge(self, outputs, targets, *args):
        # aux_edge = outputs["aux_edge"].clone()
        targets = targets['instances']
        aux_edge = outputs["aux_edge"]
        target_edge = []
        target_ignore_edge = []
        for t in targets:
            # print()
            if t["masks_sed"].shape[0] == 0:
                target_edge.append(
                    torch.zeros(1, 1, t["masks_sed"].shape[1], t["masks_sed"].shape[2], device=t["masks_sed"].device))
                target_ignore_edge.append(
                    torch.ones(1, 1, t["masks_sed"].shape[1], t["masks_sed"].shape[2], device=t["masks_sed"].device))
            else:
                target_edge.append(torch.max(t["masks_sed"].clone(), dim=0).values[None][None])
                target_ignore_edge.append(t["sed_ignore_masks"][0].clone()[None][None])
        target_edge = torch.cat(target_edge, dim=0).type(aux_edge.dtype)
        # print(target_edge.shape)
        ignore_edge = torch.cat(target_ignore_edge, dim=0).type(aux_edge.dtype)

        losses = {
            "loss_edge": batch_edge_bce_loss_jit(aux_edge, target_edge, ignore_edge)
            # "loss_sed": batch_edge_bce_loss(point_logits, point_labels, point_ignore_masks, target_label, num_masks,
            #                                     int(self.max_iters), int(self.iters))
        }
        return losses

    def loss_masks_seg(self, outputs, targets, indices, num_masks):

        assert "pred_masks_sed" in outputs
        targets = targets['instances']
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks_sed"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        losses = {
            "loss_mask_seg": sigmoid_ce_loss_jit(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss_jit(src_masks, target_masks, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_sed_non(self, outputs, targets, indices, num_masks):
        assert "pred_masks_sed" in outputs

        targets = targets['instances']
        src_idx = self._get_src_permutation_idx(indices)  # batch, pred_idx
        tgt_idx = self._get_tgt_permutation_idx(indices)  # batch, tgt_idx

        src_masks = outputs["pred_masks_sed"]
        tgt_masks = torch.zeros_like(src_masks, device=src_masks.device)
        masks = [t["masks_sed"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        tgt_masks[src_idx] = target_masks[tgt_idx].to(tgt_masks)

        tgt_label = torch.full(src_masks.shape[0:2], fill_value=self.num_classes, device=src_masks.device)
        for i in range(src_idx[0].shape[0]):
            batch = src_idx[0][i].item()
            posi = src_idx[1][i].item()
            label = targets[batch]['labels'][tgt_idx[1][i].item()]
            tgt_label[batch][posi] = label

        ignore_masks = torch.cat([tgt['sed_ignore_masks'][0][None] if tgt['sed_ignore_masks'].shape[0] else
                                  torch.zeros(src_masks.shape[-2:])[None] for tgt in targets]).to(src_masks.device)
        ignore_masks = einops.repeat(ignore_masks, 'b h w-> b c h w', c=src_masks.shape[1])

        src_masks = einops.rearrange(src_masks, 'b c h w -> (b c) (h w)')
        tgt_masks = einops.rearrange(tgt_masks, 'b c h w -> (b c) (h w)')
        ignore_masks = einops.rearrange(ignore_masks, 'b c h w -> (b c) (h w)')
        tgt_label = einops.rearrange(tgt_label, 'b c -> (b c)')

        losses = {
            'loss_sed': batch_sed_focal_loss_jit(src_masks, tgt_masks, ignore_masks, tgt_label, num_masks,
                                               self.max_iters, self.iters, no_object_loss=self.eos_coef)
        }
        del tgt_masks
        del ignore_masks
        del src_masks
        del target_masks
        return losses

    def loss_masks_sed(self, outputs, targets, indices, num_masks):
        targets = targets['instances']
        assert "pred_masks_sed" in outputs
        count = 0
        for target in targets:
            count += len(target["labels"])
        if count == 0:
            return {"loss_sed": torch.tensor(0.0, device=outputs["pred_logits"].device)}

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks_sed"]
        src_masks = src_masks[src_idx]
        masks = [t["masks_sed"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        target_label = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        ignore_masks = []
        for t in targets:
            ignore_masks.extend([t["sed_ignore_masks"][0][None] for _ in range(len(t["labels"]))])
        ignore_masks = torch.cat(ignore_masks, dim=0).to(src_masks)

        src_masks = torch.flatten(src_masks, 1)
        target_masks = torch.flatten(target_masks, 1)
        ignore_masks = torch.flatten(ignore_masks, 1)

        losses = {
            "loss_sed": batch_sed_bce_loss_jit(src_masks, target_masks, ignore_masks, target_label, num_masks,
                                               self.max_iters, self.iters)
        }

        del src_masks
        del target_masks
        return losses

    def loss_queries_constraint(self, outputs, targets, indices, **kwargs):
        assert 'pred_const_dict' in outputs, "no const_dict in predictions"
        if outputs['pred_const_dict'] is None:
            return {}
        edge_embeds = outputs['pred_const_dict']['edge_embed']  # b
        pixel_feat = outputs['pred_const_dict']['pixel_feat']  # b c h w, c=256
        targets_instances = targets['instances']
        targets = targets['gt_sed'].clone().to(pixel_feat)  # b c h w, c=n_classes
        targets = torch.where(targets == 1, 1., 0.)
        targets = F.interpolate(targets, size=pixel_feat.shape[-2:], mode='nearest')
        global_pixel = F.adaptive_avg_pool2d(pixel_feat, (1, 1)).squeeze(-1).permute(0, 2, 1)  # b 1 c
        edge_num = torch.sum(targets, dim=(-2, -1)).unsqueeze(-1) + 1e-6  # b n 1
        edge_pixel_embeds = torch.einsum('bnhw,bchw->bnc', targets, pixel_feat)
        edge_pixel_embeds = torch.div(edge_pixel_embeds, edge_num)
        edge_pixel_embeds = torch.cat((edge_pixel_embeds, global_pixel), dim=1)  # b n+1 c

        no_cls_weight = 0.1
        sim_matrixes = []
        for edge_embed, edge_pixel_embed, indice, target in zip(list(edge_embeds), list(edge_pixel_embeds), indices,
                                                                targets_instances):
            sim_matrix = cal_similarity(edge_embed, edge_pixel_embed, method='cosine') + 1
            weight = torch.ones_like(sim_matrix, device=edge_pixel_embed.device)
            src_idx = indice[0].tolist()
            tgt_cls = [target['labels'][tgt_idx].item() for tgt_idx in indice[1].tolist()]
            for x, y in zip(src_idx, tgt_cls):
                weight[x, y] = weight[x, y] * -1
            for i in range(weight.shape[0]):
                if i not in src_idx:
                    weight[i, :] = weight[i, :] * no_cls_weight
            sim_matrix = torch.mul(sim_matrix, weight)
            sim_matrixes.append(sim_matrix[None])
        sim_matrixes = torch.cat(sim_matrixes, dim=0)
        losses = {
            "loss_queries_const": torch.sum(sim_matrixes)
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks_sed': self.loss_masks_sed_non,
            'masks_seg': self.loss_masks_seg,
            'edge': self.loss_edge,
            'queries_const': self.loss_queries_constraint,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"

        return loss_map[loss](outputs, targets, indices, num_masks=num_masks)

    def forward(self, outputs, targets, cur_iter=0, **kwargs):
        self.iters = cur_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets['instances'])

        num_masks = sum(len(t["labels"]) for t in targets['instances'])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_output in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_output, targets['instances'])
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_output, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
