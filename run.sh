#!/usr/bin/env bash
export WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/5733426d-ebda-460c-b751-49380e643e41"
export PROJ_DIR="/home/ilab/dolijun/Dolijun/Yan2/sed/sed_mask_classification"
#PYTHON="/home/ilab/anaconda3/envs/pytorch1_13/bin/python"
PYTHON="/home/ilab/anaconda3/envs/py311_torch200_cuda118/bin/python"  # 使用最新环境训练速度快且精确度高
#PYTHON="/home/ilab/anaconda3/envs/py38_torch113_cuda116/bin/python"
cd ${PROJ_DIR}
# Experiments
# ${PYTHON} tools/train_sbd.py --save-dir ${SAVE}
# # orig MBTransformer: 78.6 SAVE=save/test_sbd/v256_orig
# # 去除多分支后：78.1 SAVE=save/test_sbd/v256
# # 去除一条跳跃连接：78.1 SAVE=save/test_sbd/v256_nox
# # 去除多分支，保留跳跃连击  SAVE=save/test_sbd/v256_wx

# Experiment
# # mask_classification
# # test_1:
#   SAVE=save/mask_cls_sbd/rank_100  rank_=100
#   成功收敛，边缘定位不准确

# # 相同的框架，使用pixel cls 监督，证明模型的框架是没问题的
# 实验01：SAVE=save/mask_cls_sbd/pixel_cls_01
# 直接输出类别通道的语义边缘检测 78.5
# 实验02：SAVE=save/mask_cls_sbd/pixel_cls_02
# 矩阵乘法之后输出, mask cls 的结构，pixel cls 的监督
#
# # 调整损失权重 这些结果的 rank=100
# 1. mask_cls_loss_weight_01: save/mask_cls_sbd/mask_cls_loss_weight_01
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 1.0 SED_WEIGHT: 1.0  0.777
# 实验结果的边缘可视化质量较差，可能的原因是没有一个稳定的匹配监督，在后面的实验中调整cls的权重
# 2. mask_cls_loss_weight_02: save/mask_cls_sbd/mask_cls_loss_weight_02
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 2.0 SED_WEIGHT: 1.0  0.780
# 3. mask_cls_loss_weight_03: save/mask_cls_sbd/mask_cls_loss_weight_03
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 0.5 SED_WEIGHT: 1.0  0.778
# 4. mask_cls_loss_weight_04: save/mask_cls_sbd/mask_cls_loss_weight_04
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 1.0 SED_WEIGHT: 2.0  0.780
# 注： 思考，应该将其中一个权重设置的较大，保证稳定的匹配，可以将类别权重调大，看是否可以稳定的匹配
# 5. mask_cls_loss_weight_05: save/mask_cls_sbd/mask_cls_loss_weight_05
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 5.0 SED_WEIGHT: 1.0 0.781
# 6. mask_cls_loss_weight_06: save/mask_cls_sbd/mask_cls_loss_weight_06
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 10.0 SED_WEIGHT: 1.0 0.779
# 7. mask_cls_loss_weight_07: save/mask_cls_sbd/mask_cls_loss_weight_07
# 这个实验还带了 边缘监督！！后面的09实验进行了补充
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 1.0 SED_WEIGHT: 5.0 0.500
# 8. mask_cls_loss_weight_08: save/mask_cls_sbd/mask_cls_loss_weight_08
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 1.0 SED_WEIGHT: 10.0  没有运行
# 9. mask_cls_loss_weight_08: save/mask_cls_sbd/mask_cls_loss_weight_09
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 1.0 SED_WEIGHT: 5.0  0.779
# # 注意： 上面的rank=100（ 设置错了），后面补充 rank=256的实验
# 9. mask_cls_loss_weight_05_256： SAVE=save/mask_cls_sbd/mask_cls_loss_weight_05_256
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 5.0 SED_WEIGHT: 1.0 0.780
# 10. mask_cls_loss_weight_10： SAVE=save/mask_cls_sbd/mask_cls_loss_weight_10
# 注：这个实验还是在调整损失权重，这个实验之后，调整权重的实验结束
# NO_OBJECT_WEIGHT: 0.1 CLASS_WEIGHT: 5.0 SED_WEIGHT: 2.0 0.778

# # # 匹配前缓冲
# mask_feat_embed_01：SAVE=save/mask_cls_sbd/mask_feat_embed_01
# 在sum_ 后面加一层mlp，用来缓冲 sum_ 和 buffer_ 的匹配 0.775
# mask_feat_embed_02：SAVE=save/mask_cls_sbd/mask_feat_embed_02
# 缓冲 + 边缘约束 ！！！ 注意这里的边缘约束是起到正向作用的！ 0.778
# 直接生成边缘结构后单独监督 并且权重是：CLASS_WEIGHT: 5.0 SED_WEIGHT: 1.0
# mask_feat_embed_03: SAVE=save/mask_cls_sbd/mask_feat_embed_03
# 只添加边缘约束，没有缓冲 效果很差

# # # 模型结构添加边缘约束
# 1. save/mask_cls_sbd/mask_feat_embed_temp_edge_01
# 直接生成边缘结果后和原来的边缘加在一起，效果很差 0.568
# 2. save/mask_cls_sbd/mask_feat_embed_temp_edge_02

# # 调整监督，增加正样本数量

# # queries 数量 save/mask_cls_sbd/n_query/
# n_32
# n_64
# n_128

# # deep_supervision:
# 通过对比深监督的层数，找到最优层数，用于后续网络的设计
# 注意下列实验的序号
# v04 deep=1 Mean MF-ODS:  0.766
# v00 deep=3 测试100k和95k 都为 0.774
# v01 deep=6 测试结果：0.777
# v05 deep=9
# v06 deep=12
# v02 deep=6 调整权重 --cls_weight 1.0 测试结果：0.774
# v03 deep=6 调整学习率 --base-lr 5e-6 0.772

# # 添加空监督 save/mask_cls_sbd/no_object_loss
# 对比的 baseline 是 6 层深监督，deep_supervision/v01 0.777
# v00 no_object_weight=0.1 0.781

# # 修改交叉注意力的特征 save/mask_cls_sbd/cross_feat
# baseline 是 no_object_loss/v00 side_5 0.781
# v00 buffer
# v01 buffer + side5_
# v02 side5_ + buffer
# v03 num_aux_layers 0
# v04 side5_ resize plus pixel feat + buffer
# v05 [s5, s4] + buffer
# v06 s45_

# # todo：
# 对比同一网络不同层的精度
# depth9 和 12
# 不使用深监督 + no_object_loss
# 调整 nqueries no_object_loss 20 0.1, 20 0.5, 20 1.0; 64 0.1
# hybrid match
# point rend

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d3_w0_1
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 3 --no_object_weight 0.1
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 3 --save_pred

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d6_w0_2
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --no_object_weight 0.2
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d3_w0_5
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 3 --no_object_weight 0.5
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 3 --save_pred

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d3_w1_0
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 3 --no_object_weight 1.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 3 --save_pred

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d6_w0_1_onlybce
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --no_object_bce_weight 0.1 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d6_w0_05_onlybce
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --no_object_bce_weight 0.05 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d6_bw0_1_cw0_05
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --no_object_weight 0.05 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d6_w0_02_onlybce
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --no_object_bce_weight 0.02 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

#SAVE=save_disk/mask_cls_sbd/no_object_loss/q64_d3_w0_2
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 3 --no_object_weight 0.2 --n_queries 64
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 3 --save_pred --n_queries 64
#
#SAVE=save_disk/mask_cls_sbd/no_object_loss/q64_d3_w0_5
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 3 --no_object_weight 0.5 --n_queries 64
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 3 --save_pred --n_queries 64
#
#SAVE=save_disk/mask_cls_sbd/no_object_loss/q64_d6_w1_0
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --no_object_weight 1.0 --n_queries 64
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 64

# v00: 对 pixel feat 进行边缘监督，权重0.01
#SAVE=save_disk/mask_cls_sbd/edge_superv/v00
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6  --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

## v01: 对边缘监督 loss_edge 监督，权重0.05
#SAVE=save_disk/mask_cls_sbd/edge_superv/v01
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6  --n_queries 20 --loss_edge 0.05
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

## v02: 对边缘监督 loss_edge 监督，权重0.1
#SAVE=save_disk/mask_cls_sbd/no_object_loss/q20_d6_bw1_0_cw_0_05_cls2_0
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 \
# --n_queries 20 --no_object_weight 0.05 --no_object_bce_weight 1.0 --cls_weight 1.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

### v03: 对边缘监督 loss_edge 监督，权重1.0
#SAVE=save_disk/mask_cls_sbd/edge_superv/v03
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6  --n_queries 20 --loss_edge2 1.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

#SAVE=save_disk/mask_cls_sbd/edge_superv/v04
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6  --n_queries 20 --loss_edge2 0.05
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20


#SAVE=save_disk/mask_cls_sbd/n_queries/l5
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 1 --n_queries 5
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 1 --save_pred --n_queries 5

#SAVE=save_disk/mask_cls_sbd/n_queries/l10
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --n_queries 10
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 10

#SAVE=save_disk/queries_const/init/sed_loss
##${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000_sed --rank 256 --num_aux_layers 6 --save_pred --n_queries 20
# swin
# v00 使用了 sed 监督
# v01 没有使用 sed 监督
# v02 only pixel supervision


# convnextv2
# v00 没有使用 sed 监督
# v01 使用 sed 监督
# v02 only pixel supervision
# v03 添加上采样 neck

#SAVE=save_disk/backbone/convnextv2/v01
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

#SAVE=save_disk/backbone/convnextv2/v04_upsample_nosed
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

# onl
#SAVE=save_disk/backbone/convnextv2/v02_pixel
#${PYTHON} tools/train_sbd.py --save-dir ${SAVE} --backbone convnext --pre-model ckpt/convnextv2_base_22k_384_ema.pt
#${PYTHON} tools/inference_sbd_for_test.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --backbone convnext

#SAVE=save_disk/backbone/swin/v02_pixel
#${PYTHON} tools/train_sbd.py --save-dir ${SAVE} --backbone swin_base --pre-model ckpt/swim_base_patch4_384_22k.pth
#${PYTHON} tools/inference_sbd_for_test.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --backbone swin_base

#SAVE=save_disk/pixel_feats/stage5 # all_stage einsum1_stage1_factor3
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 256 --num_aux_layers 6 --n_queries 20
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 256 --num_aux_layers 6 --save_pred --n_queries 20

# no_aca/debug_01 使用了层次监督，修改了分割监督的位置
# no_aca/debug_02 不使用层次监督,
# debug_03_BCA1  单层BCA，层次特征参与  0.780
# debug_03_BCA3 3层BCA，层次特征参与 0.775
# debug_03_BCA3_l3 3层 BCA，层次监督 0.783
# debug_03_BCA3_l3_auxSeg 3层BCA 层次监督 0.784
#

#SAVE=save_disk/seg_loss/no_seg_00
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 128 --num_aux_layers 3 --sed_weight 20.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 128 --num_aux_layers 3 --save_pred
#cp -r models ${SAVE}



#SAVE=save_disk/losses/0_noncls_nofocal_10
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 128 --num_aux_layers 3 --sed_weight 1.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 128 --num_aux_layers 3 --save_pred
#cp -r models ${SAVE}

# SAVE=save_disk/mask_head/conv1_r124_03 修改了mlp的隐藏层，78.4 
# SAVE=save_disk/mask_head/conv1_r124_04 添加了中间的层（Conv3） 更新 mask_features
# SAVE=save_disk/mask_head/conv1_r124_05 init 中间层(ConvNextBlock)

#SAVE=save_disk/mask_head/conv1_r1234_03
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 128 --num_aux_layers 3 --sed_weight 10.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 128 --num_aux_layers 3 --save_pred
#cp -r models ${SAVE}




#SAVE=save_disk/module_decoder/nl2_nq128_seg_sup
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 128 --num_aux_layers 2 --sed_weight 20.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 128 --num_aux_layers 2 --save_pred
#cp -r models ${SAVE}


#SAVE=save_disk/module_decoder/nl3_nq128_seg_sup
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 128 --num_aux_layers 3 --sed_weight 20.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 128 --num_aux_layers 3 --save_pred
#cp -r models ${SAVE}


#SAVE=save_disk/module_decoder/nl3_nq64_seg_sup
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 64 --num_aux_layers 3 --sed_weight 20.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 64 --num_aux_layers 3 --save_pred
#cp -r models ${SAVE}
#
#SAVE=save_disk/module_decoder/nl3_nq32_seg_sup
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 32 --num_aux_layers 3 --sed_weight 20.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 32 --num_aux_layers 3 --save_pred
#cp -r models ${SAVE}
#
#SAVE=save_disk/tmm_experiments/base_no_midelayer1_02
SAVE=save_disk/tmm_experiments/pixel_cls_01
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 128 --num_aux_layers 3 --sed_weight 20.0
${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-95000.pth \
          --output_folder ${SAVE}/95000 --rank 128 --num_aux_layers 3 --save_pred
cp -r models ${SAVE}
#
#SAVE=save_disk/module_decoder/nl3_nq8_seg_sup
#${PYTHON} tools/train_sbd_mask_cls.py --save-dir ${SAVE} --rank 8 --num_aux_layers 3 --sed_weight 20.0
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000 --rank 8 --num_aux_layers 3 --save_pred
#cp -r models ${SAVE}


####################################### inference for visual ##################################
#SAVE=save_disk/losses/debug_initBCA1_l3_focal_piror_pos_neg_sed10
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000_vis_th09 --rank 128 --num_aux_layers 3 --n_queries 20 \
#          --thresh ${SAVE}/100000/result_thin --visual_sed

#SAVE=save_disk/losses/a0_noncls_negpos_nofocal01
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000_vis_th06 --rank 128 --num_aux_layers 3 --n_queries 20 \
#          --thresh ${SAVE}/100000/result_thin --visual_sed

#SAVE=save_disk/losses/a0_1_noncls_nofocal_01
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000_vis --rank 128 --num_aux_layers 3 --n_queries 20 \
#          --thresh ${SAVE}/100000/result_thin --visual_sed

#SAVE=save_disk/losses/a0_noncls_nofocal_10
#${PYTHON} tools/inference_sbd_for_test_mask_cls.py --ckpt ${SAVE}/model-100000.pth \
#          --output_folder ${SAVE}/100000_vis --rank 128 --num_aux_layers 3 --n_queries 20 \
#          --thresh ${SAVE}/100000/result_thin --visual_sed

