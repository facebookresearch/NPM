# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import gzip
import json
import time
import argparse
import numpy as np

from functools import partial
from collections import Counter, defaultdict
from utils import create_blocks_from_plain_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        default="/datasets01/CC-NEWS/022719/json")
    parser.add_argument("--out_dir",
                        type=str,
                        default="train_corpus/cc_news/")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--num_shards", type=int, default=883)
    parser.add_argument("--save_flatten_data", action="store_true")
    parser.add_argument("--save_nested_data", action="store_true")

    args = parser.parse_args()

    assert args.save_flatten_data or args.save_nested_data

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    grouped_filenames = defaultdict(list)
    for fn_idx, fn in enumerate(sorted(os.listdir(args.in_dir), reverse=True)):
        date = fn.split("-")[-2][:8]
        grouped_filenames[date].append(os.path.join(args.in_dir, fn))

    filenames = list(enumerate(sorted(list(grouped_filenames.values()), reverse=True)))
    print ("We will only process %d out of %d shards (dates)" % (args.num_shards, len(filenames)))

    start_time = time.time()
    n_blocks = []

    from multiprocessing import Pool
    with Pool(min(args.num_shards, 60)) as p:
        for curr_n_blocks in p.imap(partial(process_file, args=args), filenames[:args.num_shards]):
            n_blocks.append(curr_n_blocks)
            print ("Finish processing %d/%d files (%.1fM blocks, %dmin)" % (
                len(n_blocks),
                args.num_shards,
                np.sum(n_blocks) / 1000000,
                (time.time() - start_time)/60
            ))

def process_file(pair, args):
    fn_idx, filenames = pair
    doc_id = str(fn_idx)
    lines = []
    print ("Start reading %d files for idx=%s" % (len(filenames), doc_id))
    for fn in filenames:
        with gzip.open(fn, "r") as f:
            for line in f:
                dp = json.loads(line)

                lang = dp["language"]
                if lang!="en":
                    continue

                title = dp["title"]
                text = dp["text"]

                if text is None:
                    continue

                if title is not None:
                    text = title.strip() + ". " + text.strip()

                lines.append(text)

    if len(lines)==0:
        return 0

    outputs, n_tokens = create_blocks_from_plain_text(lines, doc_idx=doc_id, max_seq_length=args.max_seq_length)
    print ("Saving %dK tokens, %d output sequences from %d text lines for idx=%s" % (
        np.sum(n_tokens)/1000, len(outputs), len(lines), doc_id))

    if args.save_flatten_data:
        out_file = os.path.join(args.out_dir, "flatten_shard{}.jsonl".format(doc_id))
        with open(out_file, "w") as f:
            for dp in outputs:
                f.write(json.dumps(dp)+"\n")

    if args.save_nested_data:
        out_file = os.path.join(args.out_dir, "BS{}_shard{}.jsonl".format(args.batch_size, doc_id))
        with open(out_file, "w") as f:
            for idx in range(len(outputs) // args.batch_size):
                curr_data = outputs[idx*args.batch_size:(idx+1)*args.batch_size]
                assert len(curr_data)==args.batch_size
                grouped_dp = {}
                for k in curr_data[0]:
                    v = [dp[k] for dp in curr_data]
                    assert len(v)==args.batch_size
                    grouped_dp[k] = v
                f.write(json.dumps(grouped_dp)+"\n")

    return len(outputs)

if __name__=='__main__':
    main()


