# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from weakref import proxy

class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath=None, filename=None, monitor=None, verbose=False,
                 save_weights_only=True, save_last=None, every: int = 100):
        super().__init__(dirpath=dirpath, filename=filename,
                         monitor=monitor, verbose=verbose,
                         save_weights_only=save_weights_only,
                         save_last=save_last)
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        #print ("on_train_batch_end called at step=%d" % pl_module.global_step)
        if pl_module.global_rank==0 and pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            trainer.save_checkpoint(current, self.save_weights_only)
            self._last_global_step_saved = trainer.global_step

            print (current)
