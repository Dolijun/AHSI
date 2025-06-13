clc; clear; close all;
path = genpath('../../matlab_from_dff');
addpath(path)
%% Evaluation on Cityscapes
categories = categories_city();

% Original GT (Thin)
eval_dir = {'/home/ilab/dlj/project/MHACA/MHACA_SED/save_city/ice_bca_mhaca/v00/180000/classes'};
result_dir = {'/home/ilab/dlj/project/MHACA/MHACA_SED/save_city/ice_bca_mhaca/v00/180000/thin_result'};

evaluation('/home/ilab/dlj/datasets/cityscapes/cityscapes-preprocess/gt_eval/gt_thin/val.mat', ...
    '/home/ilab/dlj/datasets/cityscapes/cityscapes-preprocess/gt_eval/gt_thin/inst',...
           eval_dir, result_dir, categories, 0, 99, true, 0.0035) % 0.0035



% % Original GT (Raw)
% eval_dir = {'/home/ilab/dolijun/Dolijun/Yan0/模型/hyperseg/hyperseg/save/Cityscapes_efficientnet_b1_hyperseg-m_c32_epoch350lr1e-3autoloss/inference/epoch-349/classes'};
% result_dir = {'/home/ilab/dolijun/Dolijun/Yan0/模型/hyperseg/hyperseg/save/Cityscapes_efficientnet_b1_hyperseg-m_c32_epoch350lr1e-3autoloss/inference/epoch-349/eval_result/raw'};
% evaluation('./datasets/cityscapes-preprocess/gt_eval/gt_raw/val.mat', './datasets/cityscapes-preprocess/gt_eval/gt_raw/inst',...
%            eval_dir, result_dir, categories, 0, 99, false, 0.0035) % 0.0035  SBD 0.02
% % Original GT (Raw)
% eval_dir = {'/home/ilab/dolijun/Dolijun/Yan0/模型/hyperseg/hyperseg/save/Cityscapes_efficientnet_b1_hyperseg-m_woCH_epoch350lr1e-3autoloss/inference/epoch-346/classes'};
% result_dir = {'/home/ilab/dolijun/Dolijun/Yan0/模型/hyperseg/hyperseg/save/Cityscapes_efficientnet_b1_hyperseg-m_woCH_epoch350lr1e-3autoloss/inference/epoch-346/eval_result/raw'};
% evaluation('./datasets/cityscapes-preprocess/gt_eval/gt_raw/val.mat', './datasets/cityscapes-preprocess/gt_eval/gt_raw/inst',...
%            eval_dir, result_dir, categories, 0, 99, false, 0.0035) % 0.0035  SBD 0.02

%% Evaluation on SBD
% 
% categories = categories_sbd();
% eval_dir = {'/home/ilab/dolijun/Dolijun/Yan0/模型/hyperseg/hyperseg/save/sbd_efficientnet_b1_hyperseg-m-lr5e-4/inference/epoch-6/classes'};
% result_dir = {'./eval_result/hyperseg-lr5e-4/epoch-6/gt_orig_thin'};
% 
% evaluation('./datasets/sbd-preprocess/gt_eval/gt_orig_thin/test.mat', './datasets/sbd-preprocess/gt_eval/gt_orig_thin/inst',...
%            eval_dir, result_dir, categories, 0, 99, true, 0.02); % City 0.0035 SBD 0.02

%%　TEST
% result_dir = {'/home/ilab/dolijun/Dolijun/Yan0/模型/hyperseg/hyperseg/save/Cityscapes_efficientnet_b1_hyperseg-m_epoch350lr1e-3autoloss/inference/epoch-350/raw'};
% out_file_name = '/eval_result.txt';
% out_file = [result_dir{1}, out_file_name];
% fid = fopen(out_file, 'w');
% fprintf(fid,'%s\t','hello');
% fclose(fid);




