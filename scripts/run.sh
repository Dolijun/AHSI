#!/usr/bin/env bash
set -ex

export PROJ_DIR="/root/autodl-tmp/sed/AHSI"
# PYTHON="/root/miniconda3/envs/test_env/bin/python"
PYTHON="/root/miniconda3/envs/py311_torch200_cuda118/bin/python"

cd ${PROJ_DIR}


SAVE=save/ahsi/debug
ITERS=100000
${PYTHON} -u sedtools/train.py \
--root_dir "datasets/sbd/data_proc" \
--flist_train "datasets/sbd/data_proc/trainaug_inst_orig.txt" \
--flist_val "datasets/sbd/data_proc/test_inst_orig.txt" \
--pre-model "ckpts/resnet101.pth" \
--dataset "sbd" --nclasses 20 --save-dir ${SAVE} \
--num-iters ${ITERS} --base-lr 1e-5  --crop-size 352 --batch-size 4 \
--model-name "ahsi" --backbone "resnet101"

${PYTHON} -u sedtools/inference_sed.py \
--root ${PROJ_DIR} --model-name "ahsi" --backbone "resnet101" \
--dataset "sbd" --data_root "datasets/sbd/data_proc" --file_list "datasets/sbd/data_proc/test_inst_orig.txt" \
--n_classes 20 --ckpt ${SAVE}/model-${ITERS}.pth --out_dir ${SAVE}/${ITERS}


