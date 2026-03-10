#!/bin/bash
uname -a
#date
#env
date

DATASET=refcoco+
DATA_PATH=
REFER_PATH=
BERT_PATH=
MODEL=caris
SWIN_TYPE=base
SWIN_PATH=
IMG_SIZE=448
ROOT_PATH=
OUTPUT_PATH=
SPLIT=val
now=$(date +"%Y%m%d_%H%M%S")
mkdir ${OUTPUT_PATH}
mkdir ${OUTPUT_PATH}/${NAME}
mkdir ${OUTPUT_PATH}/${NAME}/${DATASET}

python -m torch.distributed.launch --master_port 12379 train.py --model ${MODEL} \
        --dataset ${DATASET} --model_id ${DATASET} --batch-size 1 --pin_mem --print-freq 100 --workers 8 \
        --lr 1e-4 --wd 1e-2 --swin_type base \
        --warmup --warmup_ratio 1e-3 --warmup_iters 1500 --clip_grads --clip_value 1.0 \
        --pretrained_swin_weights ${SWIN_PATH} --epochs 50 --img_size ${IMG_SIZE} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} --output-dir ${OUTPUT_PATH}/${NAME} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}'/'${NAME}'/'${DATASET}'/'train-${now}.txt
