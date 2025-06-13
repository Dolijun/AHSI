%% Evaluation on Cityscapes
categories = categories_city();
% Original GT (Thin)
% eval_dir 是推理结果路径（什么集的？）
eval_dir = {'../../../exps/cityscapes/dff/dff_val/fuse';...
            '../../../exps/cityscapes/casenet/casenet_val/fuse'};
% 结果存储路径
result_dir = {'../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_thin/dff';...
              '../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_thin/casenet'};
% val.mat 存储的是验证集的文件名称
% gt_thin 是文件本身
evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_thin/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_thin',...
           eval_dir, result_dir, categories, 0, 99, true, 0.02) % 0.0035
% Original GT (Raw)
eval_dir = {'../../../exps/cityscapes/dff/dff_val/fuse';...
            '../../../exps/cityscapes/casenet/casenet_val/fuse'};
result_dir = {'../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_raw/dff';...
              '../../../exps/cityscapes/result/evaluation/test/inst/gt_orig_raw/casenet'};
evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_raw/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_raw',...
           eval_dir, result_dir, categories, 0, 99, false, 0.02) % 0.0035