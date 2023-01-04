# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import hydra
import math
import torch
import pathlib
import pickle
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from dpr_scale.utils.utils import ScriptEncoder, PathManager

from pytorch_lightning.strategies import DDPShardedStrategy, DDPStrategy
from pytorch_lightning import LightningModule
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    fp16_compress_hook,
)
from pytorch_lightning.utilities.cloud_io import load as pl_load

from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location

from dpr_scale.task.all_gather import my_all_gather, my_all_gather2

class MaskedLanguageModelingTask(LightningModule):
    def __init__(
        self,
        transform,
        query_encoder_cfg,
        context_encoder_cfg,
        optim="adamw",
        warmup_steps: int = 0,
        fp16_grads: bool = False,
        task_type: str = "softmax",
        shared_encoder: bool = True,
        contrastive_warmup_steps: int = 0,
        contrastive_maskout_same_block: bool = True,
        contrastive_negative_selection: str = None,
        sim_score_type: str = "inner",
        do_phrase: bool = True,
        pretrained_checkpoint_path: str = "",
        contrastive_context_masking_ratio=0.0):

        super().__init__()
        self.transform_conf = (
            transform.text_transform
            if hasattr(transform, "text_transform")
            else transform
        )
        self.query_encoder_cfg = query_encoder_cfg
        self.context_encoder_cfg = context_encoder_cfg
        self.shared_encoder = shared_encoder

        self.optim_conf = optim
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.warmup_steps = warmup_steps
        self.fp16_grads = fp16_grads
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.setup_done = False

        self.task_type = task_type
        assert self.task_type in ["softmax", "contrastive"]
        self.sim_score_type = sim_score_type
        assert self.sim_score_type in ["inner", "l2"]
        self.contrastive_warmup_steps = contrastive_warmup_steps
        self.contrastive_maskout_same_block = contrastive_maskout_same_block
        self.contrastive_negative_selection = contrastive_negative_selection
        self.contrastive_temperature = None

        self.do_phrase = do_phrase
        self.contrastive_context_masking_ratio = contrastive_context_masking_ratio
        torch.backends.cuda.matmul.allow_tf32 = False

    def setup(self, stage: str):
        # skip building model during test.
        # Otherwise, the state dict will be re-initialized
        if stage == "test" and self.setup_done:
            return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False

        if self.shared_encoder:
            self.encoder = hydra.utils.instantiate(self.query_encoder_cfg)
        else:
            self.query_encoder = hydra.utils.instantiate(self.query_encoder_cfg)
            self.context_encoder = hydra.utils.instantiate(self.context_encoder_cfg)

            raise NotImplementedError()

        if self.pretrained_checkpoint_path:
            checkpoint_dict = torch.load(
                PathManager.open(self.pretrained_checkpoint_path, "rb"),
                map_location=lambda s, l: default_restore_location(s, "cpu"),
            )

            state_dict = checkpoint_dict["state_dict"] if "state_dict" in checkpoint_dict else checkpoint_dict
            self.starting_global_step = checkpoint_dict["global_step"] if "global_step" in checkpoint_dict else 0
            self.load_state_dict(state_dict)
            print(f"Loaded state dict from {self.pretrained_checkpoint_path}")
        else:
            self.starting_global_step = 0
        self.setup_done = True

    def on_load_checkpoint(self, checkpoint) -> None:
        """
        This hook will be called before loading state_dict from a checkpoint.
        setup("fit") will built the model before loading state_dict
        """
        self.setup("fit")

    def on_pretrain_routine_start(self):
        if self.fp16_grads:
            self.trainer.training_type_plugin._model.register_comm_hook(
                None, fp16_compress_hook
            )

    def configure_optimizers(self):
        self.optimizer = hydra.utils.instantiate(self.optim_conf, self.parameters())
        if self.trainer.max_steps > 0:
            training_steps = self.trainer.max_steps
        else:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            training_steps = steps_per_epoch * self.trainer.max_epochs

        n_gpus = self.trainer.num_nodes * self.trainer.gpus
        print ("%d examples, n_gpus=%d, ws=%d" % (
                len(self.trainer.datamodule.train_dataloader()), n_gpus, self.trainer.world_size))
        print (
            f"Configured LR scheduler for total {training_steps} training steps, "
            f"with {self.warmup_steps} warmup steps."
        )

        def lr_lambda(current_step):
            current_step += self.starting_global_step
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(training_steps - current_step)
                / float(max(1, training_steps - self.warmup_steps)),
            )

        scheduler = LambdaLR(self.optimizer, lr_lambda)
        scheduler = {
            "scheduler": LambdaLR(self.optimizer, lr_lambda),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [self.optimizer], [scheduler]


    def _compute_simscores(self, hidden_states, all_hidden_states, score_mask=None):
        '''
        score_mask is for which token-pair is valid
            e.g., identity is excluded, attention_mask==0 is excluded
        hidden_states: [N, H]
        all_hidden_states: [M, H]
        to return: [N, M]
        '''
        if self.sim_score_type=="inner":
            scores = torch.inner(hidden_states, all_hidden_states)
        elif self.sim_score_type=="l2":
            scores = -torch.sqrt(self._square_l2_distance(hidden_states, all_hidden_states))
        else:
            raise NotImplementedError()

        if self.contrastive_temperature is None:
            self.contrastive_temperature = math.sqrt(hidden_states.shape[-1])
            print ("Setting contrastive_temperature=%s" % str(self.contrastive_temperature))

        if self.global_rank==0 and self.global_step % 1000 == 0:
            print (scores, scores.dtype)

        scores = scores / self.contrastive_temperature

        if score_mask is not None:
            scores = scores + (-1e10) * (~score_mask)

        return scores


    def training_step(self, batch, batch_idx):
        if self.task_type=="contrastive":
            loss = self._training_step_contrastive(batch, batch_idx)
        elif self.task_type=="softmax":
            loss = self._training_step_softmax(batch, batch_idx)
        else:
            raise NotImplementedError()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _training_step_softmax(self, batch, batch_idx):
        if len(batch["input_ids"].shape)==3:
            batch["input_ids"] = batch["input_ids"].reshape(-1, batch["input_ids"].shape[-1])
            batch["masked_input_ids"] = batch["masked_input_ids"].reshape(-1, batch["input_ids"].shape[-1])
            batch["attention_mask"] = batch["attention_mask"].reshape(-1, batch["attention_mask"].shape[-1])
            if "label_mask" in batch:
                batch["label_mask"] = batch["label_mask"].reshape(-1, batch["label_mask"].shape[-1])

        assert len(batch["input_ids"].shape)==\
            len(batch["masked_input_ids"].shape)==\
            len(batch["attention_mask"].shape)==2

        if self.global_step < 5:
            print ("input_ids: %s" % str(batch["input_ids"].shape))

        outputs = self.encoder({"input_ids": batch["masked_input_ids"],
                                "attention_mask": batch["attention_mask"]})

        logits = outputs.logits
        labels = batch["input_ids"]

        if "label_mask" in batch:
            label_mask = batch["label_mask"]
        else:
            label_mask = batch["input_ids"]!=batch["masked_input_ids"]

        losses = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]
        losses = losses.view(logits.size(0), logits.size(1)) * label_mask

        loss = torch.sum(losses) / torch.sum(label_mask)

        return loss

    def _do_encode(self, batch):
        assert batch["input_ids"].shape==batch["masked_input_ids"].shape==batch["attention_mask"].shape
        if "masked_attention_mask" in batch:
            assert batch["attention_mask"].shape==batch["masked_attention_mask"].shape

        if len(batch["attention_mask"].shape)==3:
            attention_mask = batch["attention_mask"].reshape(-1, batch["attention_mask"].shape[-1])
            masked_input_ids = batch["masked_input_ids"].reshape(-1, batch["masked_input_ids"].shape[-1])
            input_ids = batch["input_ids"].reshape(-1, batch["input_ids"].shape[-1])
            if "masked_attention_mask" in batch:
                assert self.do_phrase is not None
                masked_attention_mask = batch["masked_attention_mask"].reshape(-1, batch["masked_attention_mask"].shape[-1])
            else:
                masked_attention_mask = attention_mask
            if "label_mask" in batch:
                label_mask = batch["label_mask"].reshape(-1, batch["label_mask"].shape[-1])
            else:
                label_mask = input_ids!=masked_input_ids
        else:
            attention_mask = batch["attention_mask"]
            masked_input_ids = batch["masked_input_ids"]
            input_ids = batch["input_ids"]
            if "masked_attention_mask" in batch:
                assert self.do_phrase
                masked_attention_mask = batch["masked_attention_mask"]
            else:
                masked_attention_mask = attention_mask
            if "label_mask" in batch:
                label_mask = batch["label_mask"]
            else:
                label_mask = input_ids!=masked_input_ids

        query_outputs = self.encoder({"input_ids": masked_input_ids,
                                      "attention_mask": masked_attention_mask,
                                      "output_hidden_states": True})
        hidden_states = query_outputs.hidden_states[-1] # [batch_size, length, hidden]

        if self.global_step < 5:
            print ("Finish getting hidden_states (%s)" % str(hidden_states.shape))

        if self.contrastive_context_masking_ratio > 0.9 or \
                (self.contrastive_context_masking_ratio > 0.0 and \
                np.random.random() < self.contrastive_context_masking_ratio):
            # use randomly masked context
            other_hidden_states = hidden_states.clone()
            if self.global_step < 5:
                print ("Re-using masked inputs for other_hidden_states")

            # in this case, we allow retrieving from masked tokens
            # by using unmasked_input_ids for targets
            # assert torch.sum(attention_mask!=masked_attention_mask)==0
            targets = input_ids # masked_input_ids
            target_attention_mask = attention_mask # masked_attention_mask
        else:
            context_outputs = self.encoder({"input_ids": input_ids,
                                            "attention_mask": attention_mask,
                                            "output_hidden_states": True})
            other_hidden_states = context_outputs.hidden_states[-1] # [batch_size, length, hidden]
            if self.global_step < 5:
                print ("Using unmasked inputs for other_hidden_states")

            # here, targets are used for labels as well as tokens to retrieve
            targets = input_ids
            target_attention_mask = attention_mask

        if self.global_step < 5:
            print ("Finish getting other_hidden_states (%s)" % str(other_hidden_states.shape))

        return targets, target_attention_mask, masked_attention_mask, hidden_states, other_hidden_states, \
            None, label_mask

    def _training_step_contrastive(self, batch, batch_idx):
        targets, target_attention_mask, attention_mask, hidden_states, other_hidden_states, \
            labels, _label_mask = self._do_encode(batch)

        '''
        `targets` is used for tokens to retrieve as well as input_ids if single-token version
        (so they are always `input_ids`)
        `targets` is used for tokens to retrieve only if multi-token version
        (labels are calculated seperately)
        '''

        length = targets.shape[-1]
        BS = targets.shape[0] * length
        maskout = None
        maskout_func = None
        local_rank = self.global_rank
        world_size = self.trainer.world_size

        if isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)) and torch.distributed.get_world_size() > 1:
            attention_mask_to_send = target_attention_mask.detach()
            targets_to_send = targets.detach()

            in_tensor = torch.cat([
                attention_mask_to_send.unsqueeze(-1),
                targets_to_send.unsqueeze(-1),
                other_hidden_states], -1) # [bs, length, 2+hidden]

            out_tensor = [torch.zeros_like(in_tensor) for _ in range(torch.distributed.get_world_size())]
            out_tensor = my_all_gather2(out_tensor, in_tensor)
            out_tensor = torch.stack(out_tensor, 0) # [ws, bs, length, hidden+2]

            if self.contrastive_negative_selection=="half":
                H = self.trainer.world_size // 2
                out_tensor = out_tensor[:H] if self.global_rank < H else out_tensor[H:]
                local_rank = self.global_rank if self.global_rank < H else self.global_rank - H
                world_size = world_size // 2

            elif self.contrastive_negative_selection=="quarter":
                H = self.trainer.world_size // 4
                if self.global_rank < H:
                    out_tensor = out_tensor[:H]
                    local_rank = self.global_rank
                elif self.global_rank < 2*H:
                    out_tensor = out_tensor[H:2*H]
                    local_rank = self.global_rank - H
                elif self.global_rank < 3*H:
                    out_tensor = out_tensor[2*H:3*H]
                    local_rank = self.global_rank - 2*H
                else:
                    out_tensor = out_tensor[3*H:]
                    local_rank = self.global_rank - 3*H
                world_size = world_size // 4

            assert out_tensor.shape[-1]==2+other_hidden_states.shape[-1]
            all_attention_mask = out_tensor[:, :, :, 0].contiguous()
            all_targets = out_tensor[:, :, :, 1].contiguous()
            all_hidden_states = out_tensor[:, :, :, 2:].contiguous()

            all_attention_mask = all_attention_mask.reshape(-1, all_attention_mask.shape[-1])
            all_targets = all_targets.reshape(-1, all_targets.shape[-1])
            all_hidden_states = all_hidden_states.reshape(-1, all_hidden_states.shape[-2], all_hidden_states.shape[-1])

        else:
            all_attention_mask, all_targets, all_hidden_states = \
                    target_attention_mask, targets, other_hidden_states

            n_shards = 0
            local_rank = 0

        all_targets = all_targets.int()

        if self.global_step < 5:
            print ("Finish getting all_hidden_states (%s)" % str(all_hidden_states.shape))

        # compute_score_mask
        score_mask = torch.logical_and(
            attention_mask.reshape(-1).unsqueeze(-1), all_attention_mask.reshape(-1).unsqueeze(0))
        score_mask = score_mask.bool()
        if maskout is not None:
            pass
        elif self.contrastive_maskout_same_block:
            all_batch_size = attention_mask.shape[0] * world_size
            maskout = torch.eye(BS//length).bool().to(score_mask.device).unsqueeze(-1).tile((length, length)).reshape(BS, BS)
            maskout = torch.nn.functional.pad(
                maskout,
                (BS*local_rank, BS*(world_size-local_rank-1)),
                "constant",
                0)
        else:
            maskout = torch.eye(
                all_targets.shape[-1], dtype=torch.bool
            )[BS*local_rank:BS*(local_rank+1)]
        if maskout_func is not None:
            maskout = maskout_func(maskout.reshape(BS, -1, length)).reshape(BS, -1)
        maskout = ~maskout
        score_mask = torch.logical_and(score_mask, maskout)
        assert labels is None

        # from now on, we don't really need to distinguish batch size and length
        # so do reshape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        all_hidden_states = all_hidden_states.reshape(-1, all_hidden_states.shape[-1])
        targets = targets.reshape(-1)
        all_targets = all_targets.reshape(-1)

        # now, start computing labels
        if self.do_phrase:
            label_mask = _label_mask.reshape(-1)
        else:

            new_labels = torch.logical_and(
                targets.unsqueeze(-1)==all_targets.unsqueeze(0),
                score_mask)
            new_label_mask = torch.sum(new_labels, -1)>0
            labels = new_labels
            label_mask = torch.logical_and(_label_mask.reshape(-1), new_label_mask)

            self.log("label_ratio",
                    torch.sum(new_label_mask) / torch.sum(attention_mask).float()
                    )
            self.log("effective_label_ratio",
                    torch.sum(label_mask) / torch.sum(_label_mask).float()
                    )

        loss = 0
        def _get_contrastive_loss(scores, labels, score_mask=score_mask, label_mask=label_mask):
            scores = scores - torch.max(scores, 1, keepdim=True)[0]
            score_and_label_mask = torch.logical_and(score_mask, labels)
            x = torch.logsumexp(scores*labels - 1e10*(~score_and_label_mask), dim=-1) * label_mask
            tot = torch.logsumexp(scores - 1e10*(~score_mask), dim=-1) * label_mask
            nll = - x + tot
            loss = torch.sum(nll) / torch.sum(label_mask)
            return loss

        # squeeze to save memory
        valid_indices = torch.nonzero(label_mask).squeeze(-1)
        hidden_states = torch.index_select(hidden_states, 0, valid_indices)
        if not self.do_phrase:
            labels = torch.index_select(labels, 0, valid_indices)
        label_mask = torch.index_select(label_mask, 0, valid_indices)
        score_mask = torch.index_select(score_mask, 0, valid_indices)

        if torch.sum(label_mask)==0:
            pass
        elif self.do_phrase:
            targets = batch["labels"]
            assert valid_indices.shape[0]==2*targets.shape[0]
            all_targets = all_targets.reshape(-1, length)
            shifted_all_targets = [all_targets]

            for shift in range(targets.shape[-1]-1):
                shifted_all_targets.append(torch.nn.functional.pad(
                    shifted_all_targets[-1][:, 1:],
                    (0, 1),
                    "constant",
                    0))

            start_labels = targets[:, 0].unsqueeze(-1)==shifted_all_targets[0].reshape(1, -1)
            for idx in range(1, targets.shape[-1]):
                start_labels = torch.logical_and(
                    start_labels,
                    torch.logical_or(
                        targets[:, idx].unsqueeze(-1)==shifted_all_targets[idx].reshape(1, -1),
                        targets[:, idx].unsqueeze(-1)==0))

            end_labels = start_labels.clone()
            idx_vector = torch.sum(targets>0, -1)-2
            prev_end_labels = start_labels.clone()

            for shift in range(targets.shape[-1]-1):
                prev_end_labels = torch.nn.functional.pad(
                    prev_end_labels[:,:-1], (1, 0), "constant", 0)
                curr_mask = (idx_vector==shift).unsqueeze(-1)
                end_labels = end_labels * (~curr_mask) + prev_end_labels * curr_mask

            start_hidden_states, end_hidden_states = torch.unbind(
                hidden_states.reshape(-1, 2, hidden_states.shape[-1]), 1)
            score_mask, extra_score_mask = torch.unbind(
                score_mask.reshape(-1, 2, score_mask.shape[-1]), 1)
            label_mask, extra_label_mask = torch.unbind(
                label_mask.reshape(-1, 2), 1)
            assert torch.sum(score_mask!=extra_score_mask)==0
            assert torch.sum(label_mask!=extra_label_mask)==0

            start_labels = torch.logical_and(start_labels, score_mask)
            end_labels = torch.logical_and(end_labels, score_mask)

            assert torch.all(torch.any(start_labels, -1))
            assert torch.all(torch.any(end_labels, -1))

            scores_start = self._compute_simscores(
                start_hidden_states,
                all_hidden_states,
                score_mask)
            scores_end = self._compute_simscores(
                end_hidden_states,
                all_hidden_states,
                score_mask)

            if self.global_step < 5:
                print ("Finish getting scores (%s)" % str(scores_start.shape))

            loss_start = _get_contrastive_loss(scores_start, start_labels, score_mask, label_mask)
            loss_end = _get_contrastive_loss(scores_end, end_labels, score_mask, label_mask)
            self.log("contrastive_loss_start", loss_start)
            self.log("contrastive_loss_end", loss_end)
            loss += loss_start + loss_end

            start_end_simscores = torch.sum(torch.mul(start_hidden_states, end_hidden_states), -1) / self.contrastive_temperature
            assert start_end_simscores.shape==(len(start_hidden_states), )
            self.log("start_end_simscores", torch.mean(start_end_simscores))

            start_end_same = torch.all(start_labels==end_labels, -1)
            self.log("start_end_same", torch.mean(start_end_same.float()))

        else:
            scores = self._compute_simscores(hidden_states, all_hidden_states, score_mask)
            if self.global_step < 5:
                print ("Finish getting scores (%s)" % str(scores.shape))
            loss += _get_contrastive_loss(scores, labels, score_mask=score_mask, label_mask=label_mask)

        if self.global_step < 5:
            print ("step=%d, loss=%s" % (self.global_step, loss))

        return loss

class MaskedLanguageModelingEncodingTask(MaskedLanguageModelingTask):

    def __init__(self, ctx_embeddings_dir, checkpoint_path=None, use_half_precision=True,
                 remove_stopwords=False, stopwords_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.checkpoint_path = checkpoint_path
        self.use_half_precision = use_half_precision
        pathlib.Path(ctx_embeddings_dir).mkdir(parents=True, exist_ok=True)

        self.remove_stopwords = remove_stopwords

        if self.remove_stopwords:
            stopwords = set()
            #assert stopwords_dir is not None
            stopwords_dir = "/private/home/sewonmin/clean-token-retrieval/config"
            with open(os.path.join(stopwords_dir, "roberta_stopwords.txt")) as f:
                for line in f:
                    stopwords.add(int(line.strip()))
            self.stopwords = stopwords

        self.pretrained_checkpoint_path = self.checkpoint_path

    def setup(self, stage: str):
        super().setup("train")

    def _eval_step(self, batch, batch_idx):
        is_valid = batch["is_valid"]
        batch = {"input_ids": batch["input_ids"],
                 "attention_mask": batch["attention_mask"],
                 "output_hidden_states": True}

        outputs = self.encoder(batch)
        hidden_states = outputs.hidden_states[-1] # [batch_size, length, hidden]

        return batch["input_ids"].cpu(), \
            batch["attention_mask"].bool().cpu(), \
            is_valid.bool().cpu(), \
            hidden_states.half().cpu()

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        assert self.global_rank==0
        use_half_precision = self.use_half_precision

        if not self.ctx_embeddings_dir:
            self.ctx_embeddings_dir = self.trainer.weights_save_path

        embed_path = os.path.join(self.ctx_embeddings_dir,
                                  "embeddings{}.{}.npy".format(
                                      "_wo_stopwords" if self.remove_stopwords else "",
                                      "float16" if use_half_precision else "float32"))

        dstore_keys = None

        def _filter(curr_input_ids, curr_attention_mask, curr_is_valid, curr_hidden_states):
            assert len(curr_input_ids.shape)==len(curr_attention_mask.shape)==len(curr_is_valid.shape)==1
            assert len(curr_hidden_states.shape)==2
            offset = torch.sum(curr_attention_mask)
            curr_hidden_states = curr_hidden_states[:offset].numpy()

            curr_input_ids = curr_input_ids.numpy().tolist()
            curr_is_valid = curr_is_valid.numpy().tolist()

            vec = []
            for i, hidden_states in enumerate(curr_hidden_states):
                if not curr_is_valid[i]:
                    continue
                if self.remove_stopwords and curr_input_ids[i] in self.stopwords:
                    continue
                vec.append(hidden_states)
            if len(vec)==0:
                return None
            return np.stack(vec, 0)

        dstore_size = 0
        for input_ids, attention_mask, is_valid, hidden_states in tqdm(outputs):
            for curr_input_ids, curr_attention_mask, curr_is_valid, curr_hidden_states in \
                    zip(input_ids, attention_mask, is_valid, hidden_states):
                vec = _filter(curr_input_ids, curr_attention_mask, curr_is_valid, curr_hidden_states)
                if vec is not None:
                    dstore_size += len(vec)

        print ("Start saving %d embeddings at %s" % (dstore_size, embed_path))

        dstore_keys = None

        tot = 0
        for input_ids, attention_mask, is_valid, hidden_states in tqdm(outputs):
            for curr_input_ids, curr_attention_mask, curr_is_valid, curr_hidden_states in \
                    zip(input_ids, attention_mask, is_valid, hidden_states):
                vec = _filter(curr_input_ids, curr_attention_mask, curr_is_valid, curr_hidden_states)
                if vec is None:
                    continue
                if dstore_keys is None:
                    dstore_keys = np.memmap(embed_path,
                                            dtype=np.float16 if use_half_precision else np.float32,
                                            mode='w+',
                                            shape=(dstore_size, vec.shape[-1]))

                dstore_keys[tot:tot+len(vec)] = vec
                tot +=  len(vec)

        assert tot==dstore_size, (tot, dstore_size)
        print ("Finished saving %d vectors at %s" % (tot, embed_path))
        torch.distributed.barrier()  # make sure rank 0 waits for all to complete



