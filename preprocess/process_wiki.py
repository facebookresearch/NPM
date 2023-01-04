# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import time
import math
import argparse
import numpy as np

from tqdm import tqdm
from collections import defaultdict, Counter
from functools import partial

from utils import create_blocks_from_plain_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path",
                        type=str,
                        default="/checkpoint/sewonmin/data/FEVER/kilt_knowledgesource.json")
    parser.add_argument("--out_dir",
                        type=str,
                        default="train_corpus/enwiki/")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--remove_list", action="store_true")
    parser.add_argument("--num_shards", type=int, default=10)

    parser.add_argument("--shard_data", action="store_true")
    parser.add_argument("--save_flatten_data", action="store_true")
    parser.add_argument("--save_nested_data", action="store_true")

    args = parser.parse_args()

    assert args.save_flatten_data or args.save_nested_data

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    args.shard_file = os.path.join(args.out_dir, "text_shard{}.jsonl")
    args.flatten_file = os.path.join(args.out_dir, "flatten_shard{}.jsonl")
    args.nested_file = os.path.join(args.out_dir, "BS%d_shard{}.jsonl" % args.batch_size)
    args.tmp_file = "temp.jsonl"

    # shard Wikipedia
    if args.shard_data:
        shard_wiki(args)
    else:
        assert np.all([os.path.exists(args.shard_file.format(shard_idx))
                                      for shard_idx in range(args.num_shards)]), \
            "You must shard the data first. If you haven't sharded yet, specify `--shard_data`"

    # tokenize each shard in parallel

    from multiprocessing import Pool
    article_count, flatten_count, shard_count = 0, 0, 0
    with Pool(args.num_shards) as p:
        print ("Start tokenizing...")
        for _article_count, _flatten_count, _shard_count in p.imap(
                partial(save_each_shard, args=args), range(args.num_shards)):
            article_count += _article_count
            flatten_count += _flatten_count
            shard_count += _shard_count
    print ("Done with saving! article_count=%d, flatten_count=%d, shard_count=%d" % (
        article_count, flatten_count, shard_count
    ))

    if os.path.exists(args.tmp_file):
        os.remove(args.tmp_file)

def shard_wiki(args):
    np.random.seed(2022)

    all_lines = []
    cnt = 0
    print ("Starting sharding...")

    start_time = time.time()
    with open(args.in_path, "rb") as f:
        for line in f:
            dp = json.loads(line)

            if "(disambiguation)" in dp["wikipedia_title"]:
                continue

            if dp["wikipedia_title"].startswith("List of"):
                continue

            if len(dp["text"])<=1:
                continue

            if len(dp["wikipedia_title"].strip()) <= 1:
                continue

            text0 = dp["text"][0].strip()
            text1 = dp["text"][1].strip()
            if text1.startswith(text0) and (
                text1.endswith("refers to:") or text1.endswith("refer to:")
            ):
                continue

            sentences = [""] # section
            sentences_text = [False] # if the sentence contain text
            for sent in dp["text"]:
                is_plain_text = True

                if sent.startswith("Section::::"):
                    sent = sent.split("::::")[-1]
                    if len(sentences[-1])>0:
                        sentences.append("")
                        sentences_text.append(False)
                    is_plain_text = False

                if sent.startswith("External links"):
                    break

                if sent.startswith("BULLET::::"):
                    if args.remove_list:
                        continue

                    sent = sent.split("::::")[-1]
                    is_plain_text = False

                sent = sent.strip()
                if len(sent)==0:
                    continue

                sentences[-1] += " " + sent.strip()

                if is_plain_text:
                    sentences_text[-1] = True

            assert len(sentences)==len(sentences_text)

            if args.remove_list:
                new_sentences = []
                for sentence, has_text in zip(sentences, sentences_text):
                    if has_text and len(sentence.split())>=5:
                        new_sentences.append(sentence.strip())

                if np.sum([len(sentence.split()) for sentence in sentences]) < 50:
                    continue

                sentences = new_sentences

            if len(sentences)==0:
                continue

            all_lines.append(json.dumps({"title": dp["wikipedia_title"], "text": sentences}))
            cnt += 1

            if len(all_lines)==1000000:
                save_lines(all_lines, args.shard_file, args.num_shards)
                print ("Sharding... Saved %.1fM lines" % (cnt / 1000000))
                all_lines = []

    save_lines(all_lines, args.shard_file, args.num_shards)
    print ("Done with sharding: Saved %.1fM lines" % (cnt / 1000000))

def save_each_shard(shard_idx, args):
    outputs = []
    article_cnt = 0
    cnt = 0
    flatten_cnt = 0

    shard_file = args.shard_file.format(shard_idx)
    flatten_file = args.flatten_file.format(shard_idx) if args.save_flatten_data else args.tmp_file
    nested_file = args.nested_file.format(shard_idx) if args.save_nested_data else args.tmp_file

    with open(shard_file, "r") as f:
        with open(flatten_file, "w") as f_w_flatten:
            with open(nested_file, "w") as f_w:
                for line in f:
                    article_cnt += 1
                    dp = json.loads(line)
                    title = dp["title"]
                    sentences = dp["text"]
                    sentences = [title + ". " + sent for sent in sentences]

                    curr_outputs, _ = create_blocks_from_plain_text(sentences, doc_idx=title, max_seq_length=args.max_seq_length)
                    if args.save_flatten_data:
                        for o in curr_outputs:
                            flatten_cnt += 1
                            f_w_flatten.write(json.dumps(o)+"\n")

                    if args.save_nested_data:
                        outputs += curr_outputs

                        if len(outputs)<args.batch_size:
                            continue

                        curr_data = outputs[:args.batch_size]
                        assert len(curr_data)==args.batch_size
                        grouped_dp = {}
                        for k in curr_data[0]:
                            v = [dp[k] for dp in curr_data]
                            assert len(v)==args.batch_size
                            grouped_dp[k] = v
                        f_w.write(json.dumps(grouped_dp)+"\n")
                        cnt += 1

                        outputs = outputs[args.batch_size:]

    return article_cnt, flatten_cnt, cnt

def save_lines(all_lines, out_file, num_shards):
    np.random.shuffle(all_lines)
    N = math.ceil(len(all_lines) / num_shards)

    for idx in range(num_shards):
        with open(out_file.format(idx), "a+") as f:
            for line in all_lines[idx*N:(idx+1)*N]:
                f.write(line + "\n")

if __name__=='__main__':
    main()


