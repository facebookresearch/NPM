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
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from utils_span import mask_spans

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="train_corpus")
    parser.add_argument("--mr", type=float, default=0.15)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=10)

    args = parser.parse_args()

    ext = "_mr{}_p{}.jsonl".format(args.mr, args.p)

    def find_files(out_dir):
        if os.path.isdir(out_dir):
            return sorted([fn for sub_dir in os.listdir(out_dir) for fn in find_files(os.path.join(out_dir, sub_dir))])

        fn = out_dir
        if fn.split("/")[-1].startswith("BS{}_shard".format(args.batch_size)) and fn.endswith(".jsonl"):
            if fn.endswith(ext):
                return []
            if os.path.exists(fn.replace(".jsonl", ext)):
                return []
            return [fn]

        return []

    filenames = find_files(args.data_dir)
    filenames = [fn for fn in filenames if fn.split(".")[-2][-3] not in ["6", "7", "8"]]
    print ("Start span masking for %d files" % len(filenames))
    f = partial(mask_spans,
                masking_ratio=args.mr,
                p=args.p)

    tot = 0
    with Pool(min(len(filenames), 80)) as p:
        for _ in p.imap(f, filenames):
            tot += 1


if __name__=='__main__':
    main()



