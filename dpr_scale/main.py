# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.
import os
import hydra
from dpr_scale.conf.config import MainConfig

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.trainer import Trainer


"""
Sample commands:
Default: $ buck run //deeplearning/projects/dpr-scale:main

For debugging Hydra:
$ HYDRA_FULL_ERROR=1 buck run //deeplearning/projects/dpr-scale:main -- --info
"""


def cnt_files(path):
    import glob
    import subprocess
    BASE_DIR = "/checkpoint/sewonmin/data/preprocess_data"
    def _append_base_dir(path):
        if not path.startswith("/"):
            path = os.path.join(BASE_DIR, path)
        return path

    print ("counting files...")
    paths = [_append_base_dir(_path) for path in path.split("+")
                for _path in glob.glob(_append_base_dir(path))]
    for _path in paths:
        #cnt = int(subprocess.check_output("wc -l " + _path, shell=True).split()[0])
        #print ("%s\t\t%d" % (_path, cnt))
        print (_path)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    print(OmegaConf.to_yaml(cfg))
    #cnt_files(cfg.datamodule.train_path)
    #return

    if cfg.test_only:

        def do_test(ckpt_path):
            assert os.path.exists(ckpt_path), ckpt_path
            cfg.task.pretrained_checkpoint_path = ckpt_path

            task = hydra.utils.instantiate(cfg.task, _recursive_=False)
            transform = hydra.utils.instantiate(cfg.task.transform)
            datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)
            checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint_callback)
            lr_monitor = LearningRateMonitor(logging_interval='step')
            trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback, lr_monitor])

            trainer.test(
                model=task,
                ckpt_path=ckpt_path,
                verbose=True,
                datamodule=datamodule,
            )

        ckpt_path = cfg.task.pretrained_checkpoint_path
        if "+" in ckpt_path:
            segments = ckpt_path.split("+")
            assert len(segments)>=3
            for ckpt_idx in range(1, len(segments)-1):
                ckpt_path = segments[0] + segments[ckpt_idx] + segments[-1]
                do_test(ckpt_path)
        else:
            do_test(ckpt_path)

    else:
        if cfg.trainer.num_nodes>1 and not cfg.task.shared_model:
            checkpoint = os.path.join(
                "/checkpoint/sewonmin/hydra_outputs/lm_l_v3/init-false_BS-192_LR-1e-03",
                "0/lightning_logs/version_63863415/checkpoints",
                "latest-50000.ckpt")
            assert os.path.exists(checkpoint), checkpoint
            print ("Starting from %s" % str(checkpoint))
            cfg.task.query_encoder_cfg.model_path = checkpoint

            # cfg.task.optim.lr =  cfg.task.optim.lr / 10
            # print ("learning_rate", cfg.task.optim.lr)

        if cfg.datamodule.masking_ratio==0.4:
            print ("contrastive_negative_selection=half")
            cfg.task.contrastive_negative_selection="half"

        task = hydra.utils.instantiate(cfg.task, _recursive_=False)

        #assert cfg.task.model.model_path == cfg.task.transform.model_path
        transform = None #hydra.utils.instantiate(cfg.task.transform)
        datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)
        checkpoint_callback = hydra.utils.instantiate(cfg.checkpoint_callback)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback, lr_monitor])

        trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
