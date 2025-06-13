#!/usr/bin/env bash
export WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/5733426d-ebda-460c-b751-49380e643e41"
export PROJ_DIR="/media/ilab/back/dolijunc/Yan3/SedMaskClassification"
#PYTHON="/home/ilab/anaconda3/envs/pytorch1_13/bin/python"
PYTHON="/home/ilab/anaconda3/envs/py311_torch200_cuda118/bin/python"  # 使用最新环境训练速度快且精确度高
#PYTHON="/home/ilab/anaconda3/envs/py38_torch113_cuda116/bin/python"
cd ${PROJ_DIR}

set -ex

##### Cityscapes ########
#SAVE=save/maskcls_city/base_mask_cls_1e5_sedw1_clsw5_144_re3
#ITERS=160000
#${PYTHON} -u sedtools/train.py \
#--root_dir "datasets/cityscapes/data_proc" \
#--flist_train "datasets/cityscapes/data_proc/train.txt" \
#--flist_val "datasets/cityscapes/data_proc/val.txt" \
#--pre-model "ckpts/swin/swin_base_patch4_window12_384_22k.pth" \
#--dataset "cityscapes" --nclasses 19 --save-dir ${SAVE} \
#--num-iters ${ITERS} --base-lr 1e-5  --crop-size 512 --batch-size 2 \
#--model-name "only_mask_cls2" --rank 144 --num_aux_layers 3 \
#--backbone "swin_base"  --sed_weight 1.0 --mask-attn-t 0.4 --dice_sed_weight 0 --cls_weight 5.0
#${PYTHON} -u sedtools/inference_sed.py \
#--root ${PROJ_DIR} --model-name "only_mask_cls2" --backbone "swin_base" \
#--dataset "cityscapes" --data_root "datasets/cityscapes/data_proc" --file_list "datasets/cityscapes/data_proc/val.txt" \
#--n_classes 19 --ckpt ${SAVE}/model-${ITERS}.pth --out_dir ${SAVE}/${ITERS}_patch \
#--num_aux_layers 3 --rank 144 --mask-attn-t 0.4 --use_patch_pred

##### SBD ########
#SAVE=save/maskcls/base_mask_cls_1e5_aux_edge_sed_sedw2_dice1_clsw5_128
#SAVE=save/sedter/base_sedter_256_2e5_swin
SAVE=save/maskcls/internimage_mask_cls_1e5_sedw2_dice1_clsw5_144
# desc: 在83.0版本上面，添加拉 glmix 模块，部分模块连续运行2次
ITERS=100000
${PYTHON} -u sedtools/train.py \
--root_dir "datasets/sbd/data_proc" \
--flist_train "datasets/sbd/data_proc/trainaug_inst_orig.txt" \
--flist_val "datasets/sbd/data_proc/test_inst_orig.txt" \
--pre-model "ckpts/intern_image/internimage_b_1k_224.pth" \
--dataset "sbd" --nclasses 20 --save-dir ${SAVE} \
--num-iters ${ITERS} --base-lr 1e-5  --crop-size 352 --batch-size 4 \
--model-name "only_mask_cls2" --rank 144 --num_aux_layers 3 \
--backbone "internimage" --sed_weight 2.0 --mask-attn-t 0.4 --dice_sed_weight 0 --cls_weight 5.0
#--loss_seg 0 --loss_edge 0 --no_object_weight 1
# --model-name "only_mask_cls"

${PYTHON} -u sedtools/inference_sed.py \
--root ${PROJ_DIR} --model-name "only_mask_cls2" --backbone "internimage" \
--dataset "sbd" --data_root "datasets/sbd/data_proc" --file_list "datasets/sbd/data_proc/test_inst_orig.txt" \
--n_classes 20 --ckpt ${SAVE}/model-${ITERS}.pth --out_dir ${SAVE}/${ITERS} \
--rank 144 --num_aux_layers 3

# --mask-attn-t 0.4

