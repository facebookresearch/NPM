#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SAVE_DIR=$1
DO_PHRASE=$2
LR=$3
BS=$4
MR=$5
SPAN=$6
P=$7

init=true
wd=0.01
wm=4000
model_type=roberta-large
num_nodes=4
gpus=8
clip=2.0
msb=true
cmr=0.0
emp=true # make sure masked tokens have positives
ns=half #false #half # how to select negatives

SAVE_DIR=${SAVE_DIR}/LR-${LR}_BS-${BS}_MR-${MR}

if [[ $SPAN == "uniform" ]] ; then
    train_path=$(pwd)/train_corpus/cc_news/BS${BS}_batchshard0.jsonl
    train_path=${train_path}+$(pwd)/train_corpus/enwiki/BS${BS}_shard0.jsonl
    for i in {1..9} ; do \
        train_path=${train_path}+$(pwd)/train_corpus/cc_news/BS${BS}_batchshard0.jsonl
        train_path=${train_path}+$(pwd)/train_corpus/enwiki/BS${BS}_shard0.jsonl
    done
    
else
    SAVE_DIR=${SAVE_DIR}_P-${P}
    train_path=$(pwd)/train_corpus/cc_news/BS${BS}_batchshard0_mr${MR}_p${P}.jsonl
    train_path=${train_path}+$(pwd)/train_corpus/enwiki/BS${BS}_shard0_mr${MR}_p${P}.jsonl
    for i in {1..9} ; do \
        train_path=${train_path}+$(pwd)/train_corpus/cc_news/BS${BS}_batchshard0_mr${MR}_p${P}.jsonl
        train_path=${train_path}+$(pwd)/train_corpus/enwiki/BS${BS}_shard0_mr${MR}_p${P}.jsonl
    done
    if [[ $DO_PHRASE == "true" ]] ; then
        SPAN="span-merge"
    fi
fi

echo "$train_path"

HYDRA_FULL_ERROR=1 PYTHONPATH=. python dpr_scale/main.py -m \
    --config-name=lm.yaml \
    trainer.num_nodes=${num_nodes} \
    trainer.gpus=${gpus} \
    datamodule.batch_size=1 \
    task.optim.lr=${LR} \
    task.optim.weight_decay=0.01 \
    task.warmup_steps=${wm} \
    task.query_encoder_cfg.initialize=${init} \
    task.query_encoder_cfg.model_path=${model_type} \
    +task.do_phrase=${DO_PHRASE} \
    datamodule.train_path="${train_path}" \
    datamodule.val_path=null \
    datamodule.test_path=null \
    +datamodule.bidirectional=true \
    +datamodule.masking_ratio=${MR} \
    +datamodule.enforce_masking_positives=${emp} \
    +datamodule.masking=${SPAN} \
    +task.task_type=contrastive \
    +task.contrastive_maskout_same_block=${msb} \
    +task.contrastive_negative_selection=${ns} \
    +task.contrastive_context_masking_ratio=${cmr} \
    trainer.max_epochs=8 \
    trainer.gradient_clip_val=${clip} \
    trainer=slurm \
    hydra.launcher.name=${SAVE_DIR} \
    hydra.sweep.dir=${SAVE_DIR}





