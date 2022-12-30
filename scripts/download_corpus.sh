#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p corpus

if [[ $1 == "closed" ]] ; then
    # Download Wikipedia corpus (released by us)

    if [[ -f "corpus/enwiki/0.npy" ]] ; then
        echo "enwiki-0 already downloaded"
    else
        wget https://dl.fbaipublicfiles.com/NPM/enwiki-0.tar.gz -O corpus/enwiki-0.tar.gz
        tar -xf corpus/enwiki-0.tar.gz -C corpus && rm -f corpus/enwiki-0.tar.gz
        
        wget https://dl.fbaipublicfiles.com/NPM/CC-BY-SA-4.0 -O corpus/enwiki/LICENSE
    
    fi

    # Download rest of the corpus data (released by Shi et al. 2022)
    wget https://dl.fbaipublicfiles.com/NPM/corpus.tar.gz -O corpus.tar.gz
    tar -xf corpus.tar.gz -C corpus && rm -f corpus.tar.gz
    
fi

if [[ $1 == "enwiki" ]] ; then
    # Download Wikipedia corpus (released by us)
    wget https://dl.fbaipublicfiles.com/NPM/enwiki.tar.gz -O corpus/enwiki.tar.gz
    tar -xf corpus/enwiki.tar.gz -C corpus && rm -f corpus/enwiki.tar.gz

fi

if [[ $1 == "new-enwiki" ]] ; then
    # Download Wikipedia 2022 corpus (released by us)

    wget https://dl.fbaipublicfiles.com/NPM/new-enwiki.tar.gz -O corpus/new-enwiki.tar.gz
    tar -xf corpus/new-enwiki.tar.gz -C corpus && rm -f corpus/new-enwiki.tar.gz
    wget https://dl.fbaipublicfiles.com/NPM/CC-BY-SA-4.0 -O corpus/new-enwiki/LICENSE

fi

