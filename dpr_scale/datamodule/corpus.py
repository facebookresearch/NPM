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

from dpr_scale.datamodule.utils import CorpusDataset

class CorpusDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int = 2,
        val_batch_size: int = 0,  # defaults to batch_size
        test_batch_size: int = 0,  # defaults to val_batch_size
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 1
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.test_batch_size = (
            test_batch_size if test_batch_size else self.val_batch_size
        )

        self.datasets = {
            "train": CorpusDataset(train_path),
            "valid": CorpusDataset(val_path),
            "test": CorpusDataset(test_path),
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["train"],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_eval,
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
        return {"input_ids": torch.LongTensor([b["input_ids"] for b in batch]),
                "attention_mask": torch.LongTensor([b["attention_mask"] for b in batch]),
                "is_valid": torch.LongTensor([b["is_valid"] for b in batch]),
                }


