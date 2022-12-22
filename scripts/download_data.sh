#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p data

if [[ $1 == "closed" ]] ; then
    # Download evaluation datasets
    
    # SST-2 (script provided by Holtzman, West et al. 2022)

    mkdir data/sst2
    wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_dev.txt -O data/sst2/dev.tsv
    wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_test.txt -O data/sst2/test.tsv
    wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_train.txt -O data/sst2/train.tsv

    # AGN (data provided by Zhao et al. 2021)

    mkdir data/agn/
    wget https://github.com/tonyzhaozh/few-shot-learning/raw/main/data/agnews/train.csv -O data/agn/dev.csv

    # MR and CR (data provided by Gao et al. 2021)

    wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
    tar xvf datasets.tar
    mv original/mr data/mr
    mv original/cr data/cr
    mv original/subj data/subj
    rm -f datasets.tar
    rm -rf original
    
    # Download fuzzy verbalizers (released by Shi et al. 2022)
    if [[ -d "data/fuzzy_verbalizers" ]] ; then
        echo "fuzzy_verbalizers already downloaded"
    else
        gdown 1aRlqMnNyJbgkMm6vgpfEGonukuX_eZEk -O data/
        unzip data/fuzzy_verbalizers.zip -d data/ && rm -f data/fuzzy_verbalizers.zip
    fi

 
fi

if [[ $1 == "open" ]] ; then
    # LAMA (data provided by Petroni et al. 2019 and Zhong et al. 2022)

    wget https://dl.fbaipublicfiles.com/LAMA/data.zip
    unzip data.zip -d data/lama
    rm data.zip

    wget https://nlp.cs.princeton.edu/projects/optiprompt/data.tar.gz
    tar -xf data.tar.gz -C data/lama
    rm -f data.tar.gz

    python task/create_lama_uhn.py --srcdir data/lama/data/Google_RE

    rm -rf data/lama/data/ConceptNet
    rm -rf data/lama/data/Squad
    rm -rf data/lama/data/autoprompt_data
    rm -rf data/lama/data/cmp_lms_data

    # NQ (data provided by Lee et al. 2019 (re-formatted by Min et al. 2020))

    wget -P data/nq https://nlp.cs.washington.edu/ambigqa/data/nqopen-test.json
    wget -P data/nq https://nlp.cs.washington.edu/ambigqa/data/test_id2answers.json

    # KAMEL (data provided by Kalo and Fichtel, 2022)

    wget -O kamel.zip https://github.com/JanKalo/KAMEL/blob/master/data/kamel.zip?raw=true
    unzip kamel.zip -d data/kamel
    wget -P data/kamel https://raw.githubusercontent.com/JanKalo/KAMEL/master/question-templates.csv

    rm -rf kamel.zip
    rm -rf data/kamel/__MACOSX

    # Entity translation (data provided by us)
    wget https://dl.fbaipublicfiles.com/NPM/entity_translation.tar.gz -O data/entity_translation.tar.gz
    tar -xf data/entity_translation.tar.gz -C data && rm -f data/entity_translation.tar.gz

fi

if [[ $1 == "templama" ]] ; then
    # TempLAMA (changed and unchanged) released by us

    wget https://dl.fbaipublicfiles.com/NPM/templama.tar.gz -O data/templama.tar.gz
    tar -xf data/templama.tar.gz -C data && rm -f data/templama.tar.gz

fi


