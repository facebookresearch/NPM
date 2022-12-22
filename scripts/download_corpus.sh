#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p corpus

if [[ $1 == "closed" ]] ; then
    # Download Wikipedia corpus (released by us)

    if [[ -f "corpus/enwiki/0.jsonl" ]] ; then
        echo "enwiki-0 already downloaded"
    else
        wget https://dl.fbaipublicfiles.com/NPM/enwiki-0.tar.gz -O corpus/enwiki-0.tar.gz
        tar -xf corpus/enwiki-0.tar.gz -C corpus && rm -f corpus/enwiki-0.tar.gz
        
        wget https://dl.fbaipublicfiles.com/NPM/CC-BY-SA-4.0 -O corpus/enwiki/LICENSE
    
    fi

    # Download rest of the corpus data (released by Shi et al. 2022)
    
    if [[ -f "corpus/amazon/text.jsonl" ]] ; then
        echo "amazon already downloaded"
    else
        gdown 1srbYnaPQ1HVCsbyEvqBsxiq_8yiw7xqA -O corpus/
        tar -xf corpus/amazon.tar.gz -C corpus && rm -f corpus/amazon.tar.gz
    fi
    
    if [[ -f "corpus/cc_news/text.jsonl" ]] ; then
        echo "cc_news already downloaded"
    else
        gdown 15B39UQNzBc4QoxuQD-cokxksi8cPfBE5 -O corpus/
        tar -xf corpus/cc_news.tar.gz -C corpus && rm -f corpus/cc_news.tar.gz
    fi
    
    if [[ -f "corpus/imdb/text.jsonl" ]] ; then
        echo "imdb already downloaded"
    else
        gdown 1RWpkC1KpoKc35sM-Q0yPlamIHX4PeVJp -O corpus/
        tar -xf corpus/imdb.tar.gz -C corpus && rm -f corpus/imdb.tar.gz
    fi
    
    if [[ -f "corpus/subj/text.jsonl" ]] ; then
        echo "subj already downloaded"
    else
        gdown 18h9jYddkujQbIpucKuHoIEHEHxJbeKbP -O corpus/
        tar -xf corpus/subj.tar.gz -C corpus && rm -rf corpus/subj.tar.gz
    fi
    
fi

if [[ $1 == "open" ]] ; then
    # Download Wikipedia corpus (released by us)

    if [[ -f "corpus/enwiki/0.jsonl" ]] ; then
        echo "enwiki-0 already downloaded"
    else
        wget https://dl.fbaipublicfiles.com/NPM/enwiki-0.tar.gz -O corpus/enwiki-0.tar.gz
        tar -xf corpus/enwiki-0.tar.gz -C corpus && rm -f corpus/enwiki-0.tar.gz
        wget https://dl.fbaipublicfiles.com/NPM/CC-BY-SA-4.0 -O corpus/enwiki/LICENSE
    fi

    wget https://dl.fbaipublicfiles.com/NPM/enwiki.tar.gz -O corpus/enwiki.tar.gz
    tar -xf corpus/enwiki.tar.gz -C corpus && rm -f corpus/enwiki.tar.gz

fi

if [[ $1 == "templama" ]] ; then
    # Download Wikipedia 2022 corpus (released by us)

    wget https://dl.fbaipublicfiles.com/NPM/new-enwiki.tar.gz -O corpus/new-enwiki.tar.gz
    tar -xf corpus/new-enwiki.tar.gz -C corpus && rm -f corpus/new-enwiki.tar.gz
    wget https://dl.fbaipublicfiles.com/NPM/CC-BY-SA-4.0 -O corpus/new-enwiki/LICENSE

fi

