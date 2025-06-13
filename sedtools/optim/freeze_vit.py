
def freeze_vit_of_beit_adapter(extractor):
    for key, param in extractor.named_parameters():
        if 'patch_embed' in key or 'spm' in key or 'interactions' in key or 'up.' in key or 'norm1' in key \
            or 'norm2' in key or 'norm3' in key or 'norm4' in key or 'attn.relative_position_bias_table' in key:
            continue
        param.requires_grad = False
    return extractor


def freeze_vit_of_sam_adapter(extractor):
    for key, param in extractor.named_parameters():
        # if 'vit.' in key or 'blocks.' in key or 'patch_embed' in key or 'pos_embed' in key:
        if 'vit.' in key:
            param.requires_grad = False
    return extractor


def freeze_vit_of_spm(extractor):
    for key, param in extractor.named_parameters():
        # if 'vit.' in key or 'blocks.' in key or 'patch_embed' in key or 'pos_embed' in key:
        if 'stem.' in key:
            param.requires_grad = False
    return extractor





