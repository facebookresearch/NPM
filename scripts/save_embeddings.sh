#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

out=$1
corpus=$2
open=$3
bs=$4

checkpoint_path=$(pwd)/${out}/model.ckpt
ctx_embeddings_dir=$(pwd)/${out}/dstore/${corpus}

if [[ $open == "true" ]] ; then
    if [[ $corpus == "enwiki-"* ]] ; then
        
        arr=(${corpus//-/ })
        data_path=$(pwd)/corpus/enwiki/${arr[1]}.jsonl

        if [[ -f "${ctx_embeddings_dir}/embeddings_wo_stopwords.float16.npy" ]] ; then
            echo "embeddings already saved"
        else
            PYTHONPATH=. python \
                dpr_scale/generate_lm_embeddings.py -m \
                --config-name lm.yaml \
                datamodule.batch_size=$bs \
                datamodule.train_path=null \
                datamodule.val_path=null \
                datamodule.test_path=${data_path} \
                +datamodule.bidirectional=true \
                trainer.num_nodes=1 \
                trainer.gpus=1 \
                task.query_encoder_cfg.model_path=roberta-large \
                +task.ctx_embeddings_dir=${ctx_embeddings_dir} \
                +task.task_type="contrastive" \
                +task.checkpoint_path=${checkpoint_path} \
                +task.remove_stopwords=true \
                #trainer=slurm \
                #hydra.launcher.name=npm-${corpus} \
                #hydra.sweep.dir=${out} \
                #hydra.launcher.partition=devlab \
                #hydra.launcher.cpus_per_task=5
        fi
    else
        echo "corpus has to be enwiki-* (currently, ${corpus})"
        exit
    fi

else
    if [[ $corpus == "enwiki-0" ]] ; then
        data_path=$(pwd)/corpus/enwiki/0.jsonl
    else
        data_path=$(pwd)/corpus/${corpus}/text.jsonl
    fi

    if [[ -f "${ctx_embeddings_dir}/embeddings.float16.npy" ]] ; then
        echo "embeddings already saved"
    else
        PYTHONPATH=. python \
            dpr_scale/generate_lm_embeddings.py -m \
            --config-name lm.yaml \
            datamodule.batch_size=$bs \
            datamodule.train_path=null \
            datamodule.val_path=null \
            datamodule.test_path=${data_path} \
            +datamodule.bidirectional=true \
            trainer.num_nodes=1 \
            trainer.gpus=1 \
            task.query_encoder_cfg.model_path=roberta-large \
            +task.ctx_embeddings_dir=${ctx_embeddings_dir} \
            +task.task_type="contrastive" \
            +task.checkpoint_path=${checkpoint_path} \
            +task.remove_stopwords=false \
            #trainer=slurm \
            #hydra.launcher.name=npm-${corpus} \
            #hydra.sweep.dir=${out} \
            #hydra.launcher.partition=devlab \
            #hydra.launcher.cpus_per_task=5
    fi

fi
