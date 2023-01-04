#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

model_name=$1
corpus=$2
open=$3
bs=$4


out=$(pwd)/save/${model_name}
ctx_embeddings_dir=${out}/dstore/${corpus}

if [[ $open == "true" ]] ; then
    if [[ $corpus == "enwiki-"* ]] ; then
        
        arr=(${corpus//-/ })
        data_path=$(pwd)/corpus/enwiki/${arr[1]}.npy
        if [[ -f "${ctx_embeddings_dir}/embeddings_wo_stopwords.float16.npy" ]] ; then
            echo "embeddings already saved"
        else
            PYTHONPATH=. python \
                dpr_scale/generate_lm_embeddings.py -m \
                --config-name lm.yaml \
                datamodule._target_=dpr_scale.datamodule.corpus.CorpusDataModule \
                datamodule.batch_size=$bs \
                datamodule.train_path=null \
                datamodule.val_path=null \
                datamodule.test_path=${data_path} \
                trainer.num_nodes=1 \
                trainer.precision=32 \
                trainer.gpus=1 \
                task.query_encoder_cfg.model_path=facebook/${model_name} \
                +task.ctx_embeddings_dir=${ctx_embeddings_dir} \
                +task.stopwords_dir=$(pwd)/config \
                +task.task_type="contrastive" \
                +task.remove_stopwords=true \
                trainer=slurm \
                hydra.launcher.name=npm-${corpus} \
                hydra.sweep.dir=${out} \
                hydra.launcher.partition=devlab \
                hydra.launcher.cpus_per_task=5
        fi
    else
        echo "corpus has to be enwiki-* (currently, ${corpus})"
        exit
    fi

else
    if [[ $corpus == "enwiki-0" ]] ; then
        data_path=$(pwd)/corpus/enwiki/0.npy
    else
        data_path=$(pwd)/corpus/${corpus}/text.npy
    fi

    if [[ -f "${ctx_embeddings_dir}/embeddings.float16.npy" ]] ; then
        echo "embeddings already saved"
    else
        PYTHONPATH=. python \
            dpr_scale/generate_lm_embeddings.py -m \
            --config-name lm.yaml \
            datamodule._target_=dpr_scale.datamodule.corpus.CorpusDataModule \
            datamodule.batch_size=$bs \
            datamodule.train_path=null \
            datamodule.val_path=null \
            datamodule.test_path=${data_path} \
            trainer.num_nodes=1 \
            trainer.precision=32 \
            trainer.gpus=1 \
            task.query_encoder_cfg.model_path=facebook/${model_name} \
            +task.ctx_embeddings_dir=${ctx_embeddings_dir} \
            +task.task_type="contrastive" \
            +task.remove_stopwords=false \
            #trainer=slurm \
            #hydra.launcher.name=npm-${corpus} \
            #hydra.sweep.dir=${out} \
            #hydra.launcher.partition=devlab \
            #hydra.launcher.cpus_per_task=5
    fi

fi
