%% only Road
% function result_cls = benchmark_category(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist, bestT_road)
% result_img = evaluate_imgs(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist, bestT_road);
% result_cls = collect_eval_bdry(result_img);


%% All
function result_cls = benchmark_category(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist, bestT_road)
result_img = evaluate_imgs(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist, bestT_road);
result_cls = collect_eval_bdry(result_img);