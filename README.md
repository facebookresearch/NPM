# Nonparametric Masked Language Modeling

This repo contains the original implementation of the paper "[Nonparametric Masked Language Modeling](https://arxiv.org/abs/2212.01349)".

<p align="center">
  <img src="img/animation.gif" width="70%" height="70%">
</p>

```
@article{ min2022nonparametric,
    title={ Nonparametric Masked Language Modeling },
    author={ Min, Sewon and Shi, Weijia and Lewis, Mike and Chen, Xilun and Yih, Wen-tau and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    year={ 2022 }
}
```

Models are available from Huggingface Hub:hugs:! Check out [**npm**](https://huggingface.co/facebook/npm) (for phrase retrieval) and [**npm-single**](https://huggingface.co/facebook/npm-single) (for token retrieval).

**We are working on a simple demo where you can simply download all the resources and deploy on your machine. Stay tuned!**

### Updates
* **01/02/2023**: The code for training is released. See [train.md](train.md) for instructions.
* **12/22/2022**: The code for inference is released. Stay tuned for the code for training.

## Content

1. [Requirements](#requirements)
2. [Download Data](#download-data)
3. [Closed-set Experiments](#closed-set-experiments)
    * [Baselines](#baselines-on-closed-set-tasks)
    * [NPM](#npm-on-closed-set-tasks)
    * [NPM Single](#npm-single-on-closed-set-tasks)
4. [Open-set Experiments](#open-set-experiments)
    * [Baselines](#baselines-on-open-set-tasks)
    * [NPM](#npm-on-open-set-tasks)
5. [License](#license)
6. [Contact](#contact)

## Requirements

```
conda create -n npm python=3.7
conda activate npm
pip3 install -r requirements.txt --user
```

If you will use open-set tasks, make sure to install java as well.
```bash
conda install -c conda-forge openjdk
```

Note that multi-gpu inference is not supported for now.

## Download Data
Evaluation datasets and reference corpora can be downloaded via
```bash
# To run evaluation on closed-set tasks
bash scripts/download_data.sh closed
bash scripts/download_corpus.sh closed

# To run evaluation on open-set tasks
bash scripts/download_data.sh open
bash scripts/download_corpus.sh enwiki

# To run evaluation on TempLAMA (need Wikipedia 2022)
bash scripts/download_data.sh templama
bash scripts/download_corpus.sh new-enwiki
```

The corpus data is required for NPM and the retrieve-and-generate baselines. If you will only run parametric baselines, you can skip downloading the corpus.

All reference corpus files are saved under `corpus/` and evaluation datasets are saved under `data/`.

## Closed-set Experiments

#### Baselines on closed-set tasks
The following is the script for runing the RoBERTA-large baseline on all 9 datasets used in the paper.
```bash
python -m scripts.prompt \
    --checkpoint_path roberta-large \
    --eval_dataset agn+yahoo+rte+subj+sst2+mr+rt+cr+amazon \
    --save_dir save/roberta \
    --single
```

#### NPM on closed-set tasks

```bash
# To run on AGN, Yahoo and RTE:
bash scripts/save_embeddings.sh npm enwiki-0 false 320
bash scripts/save_embeddings.sh npm cc_news false 320
python -m scripts.prompt \
    --corpus_data enwiki-0+cc_news \
    --checkpoint_path npm \
    --eval_dataset agn+yahoo+rte \
    --temperature 5.0 \
    --save_dir save/npm

# To run on Subj:
bash scripts/save_embeddings.sh npm subj false 320
python -m scripts.prompt \
    --corpus_data subj \
    --checkpoint_path npm \
    --eval_dataset subj \
    --temperature 5.0 \
    --save_dir save/npm

# To run on SST-2, MR, RT, CR and Amazon:
bash scripts/save_embeddings.sh npm imdb false 320
bash scripts/save_embeddings.sh npm amazon false 320
python -m scripts.prompt \
    --corpus_data imdb+amazon \
    --checkpoint_path npm \
    --eval_dataset sst2+mr+rt+cr+amazon \
    --temperature 5.0 \
    --save_dir save/npm
```

Note that `scripts/save_embeddings.sh` takes
- model name (npm or npm-single)
- corpus name
- whether it is an open-set task (true or false)
- batch size (`320` is good for a 32gb GPU; if `trainer.precision=16` is used, `400` is good for a 32gb GPU)
as arguments. Embeddings are saved under `save/{model_name}/dstore`.

#### NPM Single on closed-set tasks

```bash
# To run on AGN, Yahoo and RTE:
bash scripts/save_embeddings.sh npm-single enwiki-0 false 320
bash scripts/save_embeddings.sh npm-single cc_news false 320
python -m scripts.prompt \
    --corpus_data enwiki-0+cc_news \
    --checkpoint_path npm-single \
    --eval_dataset agn+yahoo+rte \
    --temperature 5.0 \
    --single \
    --save_dir save/npm-single

# To run on Subj:
bash scripts/save_embeddings.sh npm-single subj false 320
python -m scripts.prompt \
    --corpus_data subj \
    --checkpoint_path npm-single \
    --eval_dataset subj \
    --temperature 5.0 \
    --single \
    --save_dir save/npm-single

# To run on SST-2, MR, RT, CR and Amazon:
bash scripts/save_embeddings.sh npm-single imdb false 320
bash scripts/save_embeddings.sh npm-single amazon false 320
python -m scripts.prompt \
    --corpus_data imdb+amazon \
    --checkpoint_path npm-single \
    --eval_dataset sst2+mr+rt+cr+amazon \
    --temperature 5.0 \
    --single \
    --save_dir save/npm-single
```

## Open-set Experiments

#### Baselines on open-set tasks

Run the following to run causal language model baselines (T5 baselines are TBA!).

```bash
python -m scripts.clm_prompt \
    --eval_dataset {lama-trex|lama-google_re|kamel|triviaqa|nq|entity_translation} \
    --model_name {j-6b|neo-1.3b|neo-2.7b|neox-20b|opt-1.3b|opt-2.7b|opt-6.7b|opt-13b|opt-30b|bloom-1b7|bloom-3b|bloom-7b1} \
    --save_dir save
```

By default, this does not use any passages from an external corpus. Specify `--ret bm25` if use BM25 passages from Wikipedia 2019, and `--ret bm25_2022` to use BM25 passages from Wikipedia 2022 (for TempLAMA).

#### NPM on open-set tasks

Please note that running open-set tasks requires around 70GB of RAM and 1.4TB of disk memory. If you want to reduce the RAM usage, you can specify `--keep_uint8` while running `python -m scripts.prompt` below, which reduces the RAM usage from 70GB to 40GB while increasing the datastore setting time. We will explore further optimizing RAM/disk usage in the future version of the code (PR is also welcome!).

```bash
# Note that this can be executed in parallel with up to 20 GPUs. In total, it takes about 10 GPU hours and 1.4TB of disk memory.
for i in {0..19} ; do
    bash scripts/save_embeddings.sh npm enwiki-${i} true 320
done

# Loading the model takes about 40min, and 70GB of RAM (specify `--keep_uint8` to reduce RAM usage to 40GB which increases the model loading time to 60-80min).
python -m scripts.prompt \
    --corpus_data enwiki \
    --checkpoint_path npm \
    --eval_dataset lama-trex+lama-google_re+kamel+triviaqa+nq+entity_translation \
    --save_dir save/npm \
    --remove_stopwords \
    --restricted \
    --open
```

To evaluate on TempLAMA, use `new-enwiki` instead of `enwiki`, and use `--eval_dataset {templama|unchanged_templama}`.

## License
NPM is CC-BY-NC 4.0 licensed.

## Contact

Please leave Github issues or contact Sewon Min `sewon@cs.washington.edu` for any questions.


<p align="center">
  <img src="img/affiliation.jpeg" width="60%" height="60%">
</p>
