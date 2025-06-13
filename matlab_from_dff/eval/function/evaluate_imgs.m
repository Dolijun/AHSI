% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% This script is used to calculate and record image-level precision recall
% for every predicted edge map in a directory
% --------------------------------------------------------

function [result_img] = evaluate_imgs(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist,bestT_road)
if (bestT_road == 0)
    flag_road = 0;
else
    flag_road = 1;
end
% num_file 应该是1
num_file = size(file_list, 1); %1
result_img = cell(num_file, 1);
parfor_progress(num_file);
parfor idx_file = 1:num_file %parfor
% % idx_file = 1:num_file %parfor
%     display(['Evaluating image ' num2str(idx_file) ' : ' file_list{idx_file,1}]); %file_list{idx_file, 1}
    edge_pred = double(imread([result_dir '/class_' num2str(idx_cls, '%d') '/' file_list{idx_file,1} '.png']))./255; %65536 %./255; %file_list{idx_file, 1}
    gt_load = load([gt_dir '/' file_list{idx_file,1} '.mat']); %file_list{idx_file, 1}
    gt_fields = fieldnames(gt_load);
    gt = gt_load.(gt_fields{1});
    edge_gt = full(double(gt.Boundaries{idx_cls}));
    
    if flag_road==1
        road_pred = double(imread([result_dir '/class_1/' file_list{idx_file,1} '.png']))./255;
        road_gt = full(double(gt.Boundaries{1}));
        temp_pred = zeros(size(edge_pred));
        temp_gt = zeros(size(edge_gt));
        temp_pred(road_pred>=bestT_road)=edge_pred(road_pred>=bestT_road);
        temp_gt(road_gt==1)=edge_gt(road_gt==1);
        edge_pred = temp_pred;
        edge_gt = temp_gt;
    end
   
    edge_pred = imresize(edge_pred, size(edge_gt));
    
    if(margin>0)
        edge_pred = edge_pred(1+margin:end-margin, 1+margin:end-margin);
        edge_gt = edge_gt(1+margin:end-margin, 1+margin:end-margin);
    end
    [thresh, cntR, sumR, cntP, sumP] = evaluate_bdry(edge_pred, edge_gt, nthresh, thinpb, maxDist);
    result_img{idx_file, 1} = [thresh, cntR, sumR, cntP, sumP];
    parfor_progress();
end
parfor_progress(0);