# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.
import os
import hydra
from dpr_scale.conf.config import MainConfig
from omegaconf import open_dict
from pytorch_lightning.trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    # Temp patch for datamodule refactoring
    #transform = hydra.utils.instantiate(cfg.task.transform)

    cfg.task._target_ = "dpr_scale.task.mlm_task.MaskedLanguageModelingEncodingTask"


    # trainer.fit does some setup, so we need to call it even though no training is done
    with open_dict(cfg):
        cfg.trainer.limit_train_batches = 0
        if "plugins" in cfg.trainer:
            cfg.trainer.pop("plugins")  # remove ddp_sharded, since it breaks during loading
    print(cfg)

    def run_test(test_path, ctx_embeddings_dir):
        cfg.datamodule.test_path = test_path
        cfg.task.ctx_embeddings_dir = ctx_embeddings_dir
        task = hydra.utils.instantiate(cfg.task, _recursive_=False)
        datamodule = hydra.utils.instantiate(cfg.datamodule) #, transform=transform)
        trainer = Trainer(**cfg.trainer)
        trainer.fit(task, datamodule=datamodule)
        trainer.test(task, datamodule=datamodule)

    if "+" in cfg.datamodule.test_path:
        test_paths = cfg.datamodule.test_path.split("+")
        ctx_embeddings_dirs = cfg.task.ctx_embeddings_dir.split("+")
    elif "\\?" in cfg.datamodule.test_path:
        assert "\\?" in cfg.task.ctx_embeddings_dir
        test_paths = [cfg.datamodule.test_path.replace("\\?", str(idx)) for idx in range(10)]
        ctx_embeddings_dirs = [cfg.task.ctx_embeddings_dir.replace("\\?", str(idx)) for idx in range(10)]
    else:
        test_paths = [cfg.datamodule.test_path]
        ctx_embeddings_dirs = [cfg.task.ctx_embeddings_dir]

    assert len(test_paths)==len(ctx_embeddings_dirs)

    for test_path, ctx_embeddings_dir in zip(test_paths, ctx_embeddings_dirs):
        run_test(test_path, ctx_embeddings_dir)

if __name__ == "__main__":
    main()
