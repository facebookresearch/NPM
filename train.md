# Training NPM

This is a guideline for training the NPM model. The training code is largely based on [facebookresearch/dpr-scale](https://github.com/facebookresearch/dpr-scale).

## Content

1. [Prepare Training Data](#prepare-training-data)
    * [Preprocessing](#preprocessing)
    * [Span Masking](#span-masking)
    * [Uniform Masking](#uniform-masking)
2. [Training](#training)
3. [Debugging locally](#debugging-locally): see this if you want to do a test run before running the entire pipeline.


## Prepare Training Data

### Preprocessing

#### Wikipedia
You need a Wikipedia file that following the format of [the KILT knowledge base](https://github.com/facebookresearch/KILT). Run
```bash
python3 preprocess/process_wiki.py \
  --in_path {a_json_file_in_kilt_format} \
  --save_nested_data \
  --shard_data
```
This will save `train_corpus/enwiki/text_shard[0-9].jsonl` (the sharded raw text files) and `train_corpus/enwiki/BS16_shard[0-9].jsonl` (preprocessed files).

#### CC News
You need CC News data in a specific format. Please see `process_file` in `preprocess/process_cc_news.py` to see the data format, or modify the function to read the data file you have.
```bash
python3 preprocess/process_cc_news.py \
  --in_dir {a_dir_containing_json_files} \
  --save_nested_data
```
This will save `train_corpus/cc_news/BS16_shard*.jsonl` (preprocessed files).

Note: by default, we are using `--batch_size 16`, which is good for training with 32GB GPUs. If you are using GPUs with smaller/larger memory, please modify it accordingly. It is highly recommended to use the largest possible batch size.

### Span Masking

To save the data with span masking, run the following:
```bash
python3 preprocess/mask_spans.py --mr 0.15 --p 0.5
```

In case of CC News, if the number of shards is larger than 10, the training script may not work. Therefore, we run the following to merge files so that the number of shards is 10.
```bash
python3 preprocess/concat_files.py --mr 0.15 --p 0.5
```

When you are done, the following files are ready to be used for training.
```bash
train_corpus
    /enwiki
        /BS16_shard[0-9]_mr0.15_p0.5.jsonl
    /cc_news
        /BS16_trsinahrd[0-9]_mr0.15_p0.5.jsonl
```


### Uniform Masking

You can optionally use uniform masking instead of span masking if you are interested in NPM-single (a variant of NPM that retrieves tokens instead of phrases). If you want to explore uniform masking, skip `preprocess/mask_spans.py`. You still need to concat files via `python3 preprocess/concat_files.py`.

When you are done, the following files are ready to be used for training.
```bash
train_corpus
    /enwiki
        /BS16_shard[0-9].jsonl
    /cc_news
        /BS16_trsinahrd[0-9].jsonl
```

## Training

To train NPM with span masking, run
```bash
bash scripts/train.sh {save_dir} true 3e-05 16 0.15 span 0.5
```
Each argument indicates save dir, whether it is a phrase retrieval model, learning rate, batch size, masking ratio, masking strategy, and p (a hyperparameter for span masking).

By default, we use 32 GPUs (4 nodes, 8 GPUs/node), each with 32GB memory. We use slurm and [hydra](https://github.com/facebookresearch/hydra) to run training. To run training with different configurations, see the command in `scripts/train.sh`.

You can use tensorboard to monitor training: `tensorboard --logdir {save_dir}`.

To train NPM-single with span masking, run
```
bash scripts/train.sh {save_dir} false 3e-05 16 0.15 span 0.5
```

To train NPM-single with uniform masking, run
```
bash scripts/train.sh {save_dir} false 3e-05 16 0.15 uniform
```

## Debugging Locally
If you want a training run on a subset of datas with one local GPU (instead of using slurm and hydra), simply run `scripts/train_debug.sh` instead of `scripts/train.sh` with the same arguments as in the [Training section](#training).

This use RoBERTA-base instead of RoBERTa-large, and can work with >=9GB GPU memory.

Note: This only uses the first shard of English Wikipedia (no CC-News), so if you have not started preprocessing and want to do a test run first, you can preprocess English Wikipedia only and keep CC-News later.


