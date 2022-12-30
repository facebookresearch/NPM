# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import mmap
import os
import time
import glob
import numpy as np
import pickle as pkl

from functools import partial

import torch
from pytorch_lightning import LightningDataModule

from dpr_scale.utils.utils import PathManager
from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

def _initialize(path, header=False, max_cnt=-1):
    mm = {}
    offset_dict = {}
    count = 0
    local_path = PathManager.get_local_path(path)
    file = open(local_path, mode="r")
    mm = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
    offset_dict[count] = mm.tell()
    if header:
        line = mm.readline()
    line = mm.readline()
    while line:
        count += 1
        offset = mm.tell()
        offset_dict[count] = offset
        line = mm.readline()
        if max_cnt > -1 and max_cnt == count:
            break
        #if count % 100000 == 0:
        #    print("Finished reading %.1fM lines from %s" % (count/1000000, path))
    return path, mm, offset_dict, count

class MemoryMappedDataset(torch.utils.data.Dataset):
    """
    A memory mapped dataset.
    """
    def __init__(self, path, header=False, max_cnt = -1):
        self.mm = {}
        self.offset_dict = {}
        self.count = 0

        if path is None:
            return

        paths = [_path for path in path.split("+") for _path in glob.glob(path)]

        print ("Start reading %d files" % len(paths))
        if len(paths)==0:
            max_cnt_per_path = 0
        else:
            max_cnt_per_path = max_cnt // len(paths) if max_cnt > -1 else -1
        start_time = time.time()

        path_idx = 0
        func = partial(_initialize, header=header, max_cnt=max_cnt_per_path)

        with ThreadPoolExecutor() as threads:
            for path, mm, offset_dict, count in threads.map(func, paths):
                self.mm[path_idx] = mm
                self.offset_dict.update({self.count + count: (path_idx, offset)
                                        for count, offset in offset_dict.items()})
                self.count += count
                print ("Finish reading %s (path_idx=%d, %.1fmin)" % (
                    "/".join(path.split("/")[-2:]),
                    path_idx,
                    (time.time()-start_time)/60))
                path_idx += 1

        print ("Final # of lines = %.3fM (%dmin)" % (
            self.count / 1000000, (time.time()-start_time)/60))

    def __len__(self):
        return self.count

    def process_line(self, line):
        return line

    def __getitem__(self, index):
        path_idx, offset = self.offset_dict[index]
        self.mm[path_idx].seek(offset)
        line = self.mm[path_idx].readline()
        return self.process_line(line)

class CorpusDataset(torch.utils.data.Dataset):
    def __init__(self, path):

        if path is None:
            self.tot = 0
        else:
            self.all_input_ids = np.load(path)
            self.block_idx_to_token_idx = np.load(path.replace(".npy", "_blocks.npy"))

            if os.path.exists(path.replace(".npy", "_valid.pkl")):
                with open(path.replace(".npy", "_valid.pkl"), "rb") as f:
                    self.valid_candidates = pkl.load(f)
                assert len(self.block_idx_to_token_idx)==len(self.valid_candidates)
            else:
                self.valid_candidates = None

            self.tot = len(self.block_idx_to_token_idx)
            self.block_idx_to_token_idx = np.concatenate(
                [self.block_idx_to_token_idx, [len(self.all_input_ids)]])

    def __len__(self):
        return self.tot

    def process_line(self, line):
        return line

    def __getitem__(self, index, max_seq_length=256):
        # start, end = self.block_idx_to_token_idx[index]
        start = self.block_idx_to_token_idx[index]
        end = self.block_idx_to_token_idx[index+1]

        input_ids = np.concatenate([self.all_input_ids[start:end], [0]*(max_seq_length-end+start)])
        attention_mask = [1]*(end-start) + [0]*(max_seq_length-end+start)

        if self.valid_candidates is not None:
            start_valid_indices, end_valid_indices = self.valid_candidates[index]
            is_valid = [i in start_valid_indices or i in end_valid_indices for i in range(max_seq_length)]
        else:
            is_valid = [i for i, _id in enumerate(input_ids) if _id not in [0, 2]]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "is_valid": is_valid}


