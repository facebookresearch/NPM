# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import mmap

import torch
from dpr_scale.transforms.lm_transform import LanguageModelingTransform
from dpr_scale.utils.utils import (
    ContiguousDistributedSampler,
    ContiguousDistributedSamplerForTest,
    PathManager,
    maybe_add_title,
)
from pytorch_lightning import LightningDataModule

from dpr_scale.datamodule.utils import MemoryMappedDataset

class LanguageModelingJsonlDataModule(LightningDataModule):
    """
    This reads a jsonl file with json objects from the original DPR data obtained from
    https://github.com/facebookresearch/DPR/blob/master/data/download_data.py.
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int = 2,
        val_batch_size: int = 0,  # defaults to batch_size
        test_batch_size: int = 0,  # defaults to val_batch_size
        bidirectional: bool = True,
        masking: str = None,
        masking_ratio: float = 0.0,
        enforce_masking_positives: bool = False,
        num_workers: int = 0,  # increasing this bugs out right now
        max_cnt: int = -1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.test_batch_size = (
            test_batch_size if test_batch_size else self.val_batch_size
        )

        _path = train_path if train_path is not None else test_path

        self.dpr_transform = LanguageModelingTransform(
            bidirectional=bidirectional,
            masking=masking,
            masking_ratio=masking_ratio,
            enforce_masking_positives=enforce_masking_positives
        )
        self.num_workers = num_workers
        self.datasets = {
            "train": MemoryMappedDataset(train_path, max_cnt=max_cnt),
            "valid": MemoryMappedDataset(val_path, max_cnt=max_cnt),
            "test": MemoryMappedDataset(test_path, max_cnt=max_cnt),
        }

    def train_dataloader(self):
        sampler = None
        if (
            self.trainer
            and hasattr(self.trainer, "world_size")
            and self.trainer.world_size > 1
        ):
            sampler = ContiguousDistributedSampler(
                self.datasets["train"], num_replicas_per_node=self.trainer.gpus
            )

        return torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_train,
            sampler=sampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["valid"],
            shuffle=False,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_test,
        )

    def collate_eval(self, batch):
        return self.collate(batch, "eval")

    def collate_test(self, batch):
        return self.collate(batch, "test")

    def collate_train(self, batch):
        return self.collate(batch, "train")

    def collate(self, batch, stage):
        #print ("datamodule collate called")
        return self.dpr_transform(batch, stage)

