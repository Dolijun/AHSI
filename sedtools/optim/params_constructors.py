import json

from torch.optim import AdamW


def get_num_layer_for_vit(var_name, num_max_layer):
    if not var_name.startswith('backbone.'):
        var_name = 'backbone.' + var_name
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed', 'backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks') or var_name.startswith(
            'backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


# def parse_group_params_and_lr_scales(params):
#     group_params = []
#     lr_scales = []
#     for key, value in params.items():
#         group_params.append(
#             {'params': value['params'], 'lr': value['lr'],
#              'lr_scale': value['lr_scale'],
#              'weight_decay': value['weight_decay']}
#         )
#         lr_scales.append(value['lr_scale'])
#     return group_params, lr_scales


def add_vit_layer_decay_params(
        params,
        module,
        num_layers=24,
        layer_decay_rate=0.9,
        base_lr=1e-5,
        weight_decay=0.01
):
    parameter_groups = {}

    num_layers = num_layers + 2
    layer_decay_rate = layer_decay_rate
    print('Build LayerDecayOptimizerConstructor %f - %d' %
          (layer_decay_rate, num_layers))
    # weight_decay = self.base_wd
    # lr = base_lr

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in (
                'pos_embed', 'cls_token', 'visual_embed'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        layer_id = get_num_layer_for_vit(name, num_layers)
        group_name = 'layer_%d_%s' % (layer_id, group_name)

        if group_name not in parameter_groups:
            scale = layer_decay_rate ** (num_layers - layer_id - 1)
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    to_display = {}
    for key in parameter_groups:
        to_display[key] = {
            'param_names': parameter_groups[key]['param_names'],
            'lr_scale': parameter_groups[key]['lr_scale'],
            'lr': parameter_groups[key]['lr'],
            'weight_decay': parameter_groups[key]['weight_decay'],
        }
    print('Param groups = %s' % json.dumps(to_display, indent=2))

    # state_dict = module.state_dict()
    # for group_name in parameter_groups:
    #     group = parameter_groups[group_name]
    #     for name in group["param_names"]:
    #         group["params"].append(state_dict[name])
    params.extend(parameter_groups.values())
    # return params


def add_decoder_params(
        params,
        module,
        base_lr=1e-5,
        weight_decay=0.01,
        lr_scale=1.,
):
    parameter_groups = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        group_name = f'decoder_{group_name}'

        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': lr_scale,
                'group_name': group_name,
                'lr': lr_scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    to_display = {}
    for key in parameter_groups:
        to_display[key] = {
            'param_names': parameter_groups[key]['param_names'],
            'lr_scale': parameter_groups[key]['lr_scale'],
            'lr': parameter_groups[key]['lr'],
            'weight_decay': parameter_groups[key]['weight_decay'],
        }
    print('Param groups = %s' % json.dumps(to_display, indent=2))

    # state_dict = module.state_dict()
    # for group_name in parameter_groups:
    #     group = parameter_groups[group_name]
    #     for name in group["param_names"]:
    #         group["params"].append(state_dict[name])
    params.extend(parameter_groups.values())

def add_swin_params(
        params,
        module,
        base_lr=1e-5,
        weight_decay=0.01,
        lr_scale=1.
):
    parameter_groups = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in (
                'pos_embed', 'cls_token', 'visual_embed'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': lr_scale,
                'group_name': group_name,
                'lr': lr_scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    to_display = {}
    for key in parameter_groups:
        to_display[key] = {
            'param_names': parameter_groups[key]['param_names'],
            'lr_scale': parameter_groups[key]['lr_scale'],
            'lr': parameter_groups[key]['lr'],
            'weight_decay': parameter_groups[key]['weight_decay'],
        }
    print('Param groups = %s' % json.dumps(to_display, indent=2))
    params.extend(parameter_groups.values())

def add_vit_bimla_params(
        args,
        params,
        module,
        base_lr=1e-5,
        weight_decay=0.01,
        lr_scale=1.
):
    parameter_groups = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            print('no grad: ', name)
            continue  # frozen weights

        # if 'vit.' in name or '.blocks.' in name or 'patch_embed.' in name or 'pos_embed' in name:
        if 'mla.' in name:
            cur_lr_scale = 5.
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token', 'visual_embed'):
                group_name = 'vit_mla_no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'vit_mla_decay'
                this_weight_decay = weight_decay
        else:
            cur_lr_scale = 1.
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token', 'visual_embed'):
                group_name = 'vit_no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'vit_decay'
                this_weight_decay = weight_decay

        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': cur_lr_scale,
                'group_name': group_name,
                'lr': cur_lr_scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    to_display = {}
    for key in parameter_groups:
        to_display[key] = {
            'param_names': parameter_groups[key]['param_names'],
            'lr_scale': parameter_groups[key]['lr_scale'],
            'lr': parameter_groups[key]['lr'],
            'weight_decay': parameter_groups[key]['weight_decay'],
        }
    print('Param groups = %s' % json.dumps(to_display, indent=2))
    params.extend(parameter_groups.values())


def add_sam_adapter_params(
        args,
        params,
        module,
        base_lr=1e-5,
        weight_decay=0.01,
        lr_scale=1.
):
    parameter_groups = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            print('no grad: ', name)
            continue  # frozen weights

        # if 'vit.' in name or '.blocks.' in name or 'patch_embed.' in name or 'pos_embed' in name:
        if 'vit.' in name:
            cur_lr_scale = 1.
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token', 'visual_embed'):
                group_name = 'vit_no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'vit_decay'
                this_weight_decay = weight_decay
        elif 'spm.' in name:
            # print(name)
            # print(args.spm_name)spm_name
            cur_lr_scale = 1. if args.spm_name is not None else 5.
            # print(cur_lr_scale)
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token', 'visual_embed'):
                group_name = 'spm_no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'spm_decay'
                this_weight_decay = weight_decay
        elif 'rein.' in name:
            cur_lr_scale = 5.
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token', 'visual_embed'):
                group_name = 'rein_no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'rein_decay'
                this_weight_decay = weight_decay
        else:
            cur_lr_scale = 5.
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token', 'visual_embed'):
                group_name = 'adapter_no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'adapter_decay'
                this_weight_decay = weight_decay

        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': cur_lr_scale,
                'group_name': group_name,
                'lr': cur_lr_scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    to_display = {}
    for key in parameter_groups:
        to_display[key] = {
            'param_names': parameter_groups[key]['param_names'],
            'lr_scale': parameter_groups[key]['lr_scale'],
            'lr': parameter_groups[key]['lr'],
            'weight_decay': parameter_groups[key]['weight_decay'],
        }
    print('Param groups = %s' % json.dumps(to_display, indent=2))
    params.extend(parameter_groups.values())

def add_upernet_params(
        params,
        module,
        base_lr=1e-5,
        weight_decay=0.01,
        lr_scale=1.
):
    parameter_groups = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in (
                'pos_embed', 'cls_token', 'visual_embed'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = weight_decay

        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': lr_scale,
                'group_name': group_name,
                'lr': lr_scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    to_display = {}
    for key in parameter_groups:
        to_display[key] = {
            'param_names': parameter_groups[key]['param_names'],
            'lr_scale': parameter_groups[key]['lr_scale'],
            'lr': parameter_groups[key]['lr'],
            'weight_decay': parameter_groups[key]['weight_decay'],
        }
    print('Param groups = %s' % json.dumps(to_display, indent=2))
    params.extend(parameter_groups.values())



if __name__ == '__main__':
    import torch

    # ckpt = torch.load("D:\\Download\\upernet_beit_adapter_large_640_160k_ade20k.pth.tar")
    ckpt = torch.load("/media/ilab/back/dolijunc/Yan3/SedMaskClassification/ckpts/vit_adapter/upernet_beit_adapter_large_640_160k_ade20k.pth.tar")
    print("hello_world")
    params = list()
    # add_vit_layer_decay_params(params, )
    from models.backbone.vit_adapter.beit_adapter import beit_adapter_large
    extractor = beit_adapter_large()
    add_vit_layer_decay_params(params, extractor, layer_decay_rate=0.9, base_lr=1e-5, weight_decay=0.01)
    from models.heads.upernet import uper_head
    model = uper_head()
    # model = torch.nn.DataParallel(model)
    add_decoder_params(params, model, base_lr=1e-5, weight_decay=0.01)
    # parse_group_params_and_lr_scales(params)
    # for param in params:
    #     param['params'] = len(param['params'])
    #     print(param)

    optimizer = torch.optim.AdamW(
        params, lr=1e-5, weight_decay=0.01, betas=(0.9, 0.99)
    )
    from schedulers import LrSchedulerWithWarmup
    scheduler = LrSchedulerWithWarmup(
        optimizer, max_iters=100000, warmup_steps=10000, anneal_strategy='cos', base_lr=1e-5, eta_min=1e-8
    )


    x_values = list()
    y_values = list()
    for i in range(100000):
        x_values.append(i)
        scheduler.step(i)
        y_values.append(optimizer.param_groups[0]['lr'])

    print("max_lr", max(y_values))
    print("min_lr", min(y_values))
    from matplotlib import pyplot as plt
    y_values = [y * 1e8 for y in y_values]
    plt.plot(x_values, y_values)
    # plt.plot(x_values, y_values, marker='o')
    plt.grid(True)
    plt.show()



    # params = [ params for param in params]

    # print(params)


    # print("hello")

