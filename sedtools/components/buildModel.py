import torch

def build_model(args, feat_chans):
    if args.model_name == "aca_mask_cls":
        from models import ACA
        model = ACA(nclasses=args.nclasses, rank_=args.rank, n_queries=args.n_queries,
                    num_aux_layers=args.num_aux_layers,
                    feat_chans=feat_chans)
        model = torch.nn.DataParallel(model).to(args.device)
    elif args.model_name == "ahsi":
        from models import Top_Down
        model = Top_Down(nclasses=args.nclasses, feat_chans=feat_chans, inter_channel=256)
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        raise ValueError(f"Error: unknown model_name when build model {args.model_name}")
    aux_model = None
    aux_models = None

    if aux_model is not None:
        return model, aux_model
    elif aux_models is not None:
        return model, aux_models
    else:
        return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default='vit_bimla_large')
    parser.add_argument("--model-name", type=str, default="bimla")
    parser.add_argument("--aux-head", type=str, default="bimla_auxheads")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--nclasses", type=int, default=20)

    args = parser.parse_args()

    from sedtools.components.buildExtractor import build_extractor
    extractor, feat_chans = build_extractor(args)
    model, aux_models = build_model(args, feat_chans)

    inputs = torch.randn((2, 3, 352, 352)).to(args.device)

    feats = extractor(inputs)

    results = model(feats)

    aux_results = list()
    for aux_model in aux_models:
        aux_result = aux_model(feats)
        aux_results.append(aux_result)

    print([result.shape for result in results])
    print([result.shape for result in aux_results])
    print("hello world")



