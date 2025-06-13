import torch
import collections
import torch.nn.functional as F
import math

def decoder_parameters(model):
    parameters_id = {}
    ## MHACA
    for key, v in model.named_parameters():
        if 'head' in key:
            if 'weight' in key:
                if 'score.weight' not in parameters_id:
                    parameters_id['score.weight'] = []
                parameters_id['score.weight'].append(v)
            elif 'bias' in key:
                if 'score.bias' not in parameters_id:
                    parameters_id['score.bias'] = []
                parameters_id['score.bias'].append(v)
        elif '.crossfusion' in key or '.buffer_' in key or '.BCA' in key or '.sque' in key \
                or '.ham' in key or '.mbt' in key or '.down' in key or \
                '.FuseGFF' in key or '.t' in key or '.c' in key or '.global_context' in key \
                or 'mlp' in key or 'decoder_norm' in key or 'embed' in key or '.ocr' in key or \
                'sed2edge' in key or '_embed' in key or '.aux_layers' in key or '.infer_cross_attn' in key \
                or 'rep_align.' in key or 'edge_pred.' in key or 'neck' in key or 'kernel_' in key or 'module.mask_feat' in key:
            if 'weight' in key:
                if 'br_conv.weight' not in parameters_id:
                    parameters_id['br_conv.weight'] = []
                parameters_id['br_conv.weight'].append(v)
            elif 'bias' in key:
                if 'br_conv.bias' not in parameters_id:
                    parameters_id['br_conv.bias'] = []
                parameters_id['br_conv.bias'].append(v)
            # elif 'buffer_' in key:
            #     parameters_id['br_conv.weight'].append(v)
        else:
            print(key)
            print('have other key not in parameters_id!')

    return parameters_id


def model_parameters(extractor, model):
    parameters_id = {}
    # 对于 swintransformer，把所有的参数都加入其中
    for key, v in extractor.named_parameters():
        if "module" in key:
            if ".weight" in key:
                if 'backbone.parameters' not in parameters_id:
                    parameters_id['backbone.parameters'] = []
                parameters_id['backbone.parameters'].append(v)
    ## decoder
    parameters_id.update(decoder_parameters(model))

    return parameters_id


def adjust_learning_rate(optimizer, cur_lr):
    for i in range(len(optimizer.param_groups)):
        param_group = optimizer.param_groups[i]
        if i == 0:
            param_group['lr'] = cur_lr * 1
        elif i == 1:
            param_group['lr'] = cur_lr * 5
        elif i == 2:
            param_group['lr'] = cur_lr * 5
        elif i == 3:
            param_group['lr'] = cur_lr * 5
        elif i == 4:
            param_group['lr'] = cur_lr * 5
        else:
            print('have other param not update!')


def model_parameters_resnet101(extractor, model):
    parameters_id = {}
    for key, v in extractor.named_parameters():
        if '.conv' in key or '.downsample' in key:
            if 'weight' in key:
                if 'res_conv.weight' not in parameters_id:
                    parameters_id['res_conv.weight'] = []
                parameters_id['res_conv.weight'].append(v)
        elif '.bn' in key:
            if 'weight' in key:
                if 'res_bn.weight' not in parameters_id:
                    parameters_id['res_bn.weight'] = []
                parameters_id['res_bn.weight'].append(v)
            elif 'bias' in key:
                if 'res_bn.bias' not in parameters_id:
                    parameters_id['res_bn.bias'] = []
                parameters_id['res_bn.bias'].append(v)
        else:
            print('have other key not in parameters_id!')

    ## decoder
    parameters_id.update(decoder_parameters(model))

    return parameters_id


def adjust_learning_rate_resnet101(optimizer, cur_lr):
    for i in range(len(optimizer.param_groups)):
        param_group = optimizer.param_groups[i]
        if i == 0:
            param_group['lr'] = cur_lr * 1
        elif i == 1:
            param_group['lr'] = cur_lr * 0
        elif i == 2:
            param_group['lr'] = cur_lr * 0
        elif i == 3:
            param_group['lr'] = cur_lr * 5
        elif i == 4:
            param_group['lr'] = cur_lr * 5
        elif i == 5:
            param_group['lr'] = cur_lr * 5
        elif i == 6:
            param_group['lr'] = cur_lr * 5
        else:
            print('have other param not update!')


def freeze_batchnorm(modules):
    if isinstance(modules, list):
        for module in modules:
            for m in module.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    # 冻结主干网络的BN层
                    m.eval()
    else:
        for m in modules.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                # 冻结主干网络的BN层
                m.eval()


def ckpt_pre_suffix(ckpt, model_prefix=None, model_suffix=None):
    model_prefix = "" if model_prefix is None else model_prefix + '.'
    model_suffix = "" if model_suffix is None else '.' + model_suffix
    new_ckpt = collections.OrderedDict()
    for k, v in ckpt.items():
        new_ckpt[model_prefix + k + model_suffix] = v

    return new_ckpt

