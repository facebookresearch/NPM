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

