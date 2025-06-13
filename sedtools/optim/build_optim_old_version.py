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

    # 默认 AdamW
    from torch.optim import AdamW
    optimizer = AdamW(
        params=group_params,
        lr=args.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    return optimizer
