import torch
def build_extractor(args):
    if args.backbone == "resnet101":
        from models.backbone import ResNet101
        extractor = ResNet101()
        feat_chans = None
    elif args.backbone == "convnextv2":
        from models.backbone import ConvNeXtV2
        extractor = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        feat_chans = [128, 256, 512, 1024]
    elif args.backbone == "swin_base":
        from models.backbone import SwinBackbone
        extractor = SwinBackbone(swin_type='base')
        feat_chans = [128, 256, 512, 1024]
    elif args.backbone == "swin_small":
        from models.backbone import SwinBackbone
        extractor = SwinBackbone(swin_type='small')
        feat_chans = [96, 192, 384, 768]
    elif args.backbone == "swin_tiny":
        from models.backbone import SwinBackbone
        extractor = SwinBackbone(swin_type='tiny')
        feat_chans = [96, 192, 384, 768]
    elif args.backbone == "swin_base_upernet":
        from models.backbone import SwinBackbone
        extractor = SwinBackbone(swin_type="base_for_upernet")
        feat_chans = [128, 256, 512, 1024]
    else:
        raise ValueError(f"Unknown backbone {args.backbone}")
    extractor = torch.nn.DataParallel(extractor).to(args.device)
    return extractor, feat_chans
