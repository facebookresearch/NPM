# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import time
import numpy as np
import json

def concat_files(filenames, out_file):
    start_time = time.time()
    n_files = 0
    n_lines = 0
    assert not os.path.exists(out_file)
    print ("Starting %s" % out_file)
    with open(out_file, "a+") as f_w:
        for filename in filenames:
            if not os.path.exists(filename):
                continue
            with open(filename, "r") as f:
                for line in f:
                    f_w.write(line)
                    n_lines += 1
            n_files += 1
    print ("Finish saving %d lines at %s from %d files (%dmin)" % (
        n_lines, out_file, n_files, (time.time()-start_time)/60))
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)
    print ("Finish deleting %d files (%dmin)" % (n_files, (time.time()-start_time)/60))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="train_corpus")
    parser.add_argument("--mr", type=float, default=None)
    parser.add_argument("--p", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=10)

    args = parser.parse_args()

    if args.mr is None and args.p is None:
        ext = ".jsonl"
    else:
        ext = "_mr{}_p{}.jsonl".format(args.mr, args.p)

    def find_files(out_dir):
        if os.path.isdir(out_dir):
            return sorted([fn for sub_dir in os.listdir(out_dir) for fn in find_files(os.path.join(out_dir, sub_dir))])

        fn = out_dir
        if fn.split("/")[-1].startswith("BS{}_shard".format(args.batch_size)) and fn.endswith(ext):
            return [fn]
        return []

    filenames = find_files(os.path.join(args.data_dir, "cc_news"))
    n_files_per_shard = math.ceil(len(filenames) / args.num_shards)
    for batch_idx in range(args.num_shards):
        curr_filenames = filenames[batch_idx*n_files_per_shard:(batch_idx+1)*n_files_per_shard]
        concat_files(curr_filenames,
                     os.path.join(args.data_dir, "cc_news",
                                  "BS{}_batchshard{}".format(args.batch_size, batch_idx) + ext))
    print ("Finish %d batches" % args.num_shards)


if __name__=='__main__':
    main()
