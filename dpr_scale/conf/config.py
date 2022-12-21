# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from dataclasses import dataclass, field
from typing import List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

defaults = [
    "_self_",
    {"task": "dpr"},
    # Model
    #{"task/model": "hf_model"},
    {"task/query_encoder_cfg": "hf_encoder"},
    {"task/context_encoder_cfg": "hf_encoder"},
    # Transform
    {"task/transform": "hf_transform"},
    # Optim
    {"task/optim": "adamw"},
    # Data
    {"datamodule": "default"},
    # Trainer
    {"trainer": "gpu_1_host"},
    # Trainer callbacks
    {"checkpoint_callback": "default"},
]


@dataclass
class MainConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task: Any = MISSING
    datamodule: Any = MISSING
    trainer: Any = MISSING
    test_only: bool = False
    checkpoint_callback: Any = MISSING

cs = ConfigStore.instance()

cs.store(name="config", node=MainConfig)
