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
RESUME_PATH=
OUTPUT_PATH=
SPLIT=val

mkdir ${OUTPUT_PATH}
mkdir ${OUTPUT_PATH}/${DATASET}
python eval.py --model ${MODEL} --swin_type ${SWIN_TYPE} \
        --dataset ${DATASET} --split ${SPLIT} \
        --pretrained_swin_weights ${SWIN_PATH} \
        --img_size ${IMG_SIZE} --resume ${RESUME_PATH} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}/eval-${SPLIT}.txt