# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import GPT2LMHeadModel, GPT2Config, RobertaForMaskedLM, RobertaConfig

class Encoder(nn.Module):
    def __init__(
        self,
        model_path: str = "gpt2-large",
        initialize: bool = True,
        dropout: float = 0.1,
        num_hidden_layers = None,
        hidden_size = None,
        vocab_size = None,
        use_pre_ffn_hidden_states = False,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)

        if model_path.startswith("gpt2") or model_path.startswith("/checkpoint/sewonmin/hydra_outputs/lm_"):
            if model_path.startswith("gpt2"):
                cfg = GPT2Config.from_pretrained(local_model_path)
            elif model_path.startswith("/checkpoint/sewonmin/hydra_outputs/lm_l"):
                cfg = GPT2Config.from_pretrained("gpt2-large")
            else:
                raise NotImplementedError()
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
            if num_hidden_layers is not None:
                cfg.n_layers = num_hidden_layers
            if hidden_size is not None:
                cfg.hidden_size = hidden_size
            if vocab_size is None:
                cfg.vocab_size = vocab_size
            if use_pre_ffn_hidden_states:
                from dpr_scale.models.modeling_gpt2 import MyGPT2Model
                Model = MyGPT2Model
            else:
                Model = GPT2LMHeadModel
        elif model_path.startswith("roberta"):
            cfg =RobertaConfig.from_pretrained(local_model_path)
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
            if num_hidden_layers is not None:
                cfg.num_hidden_layers = num_hidden_layers
            if hidden_size is not None:
                cfg.hidden_size = hidden_size
            if vocab_size is not None:
                cfg.vocab_size = vocab_size
            Model = RobertaForMaskedLM
        else:
            raise NotImplementedError()

        if initialize:
            self.transformer = Model.from_pretrained(local_model_path, config=cfg)
            print ("Initializing from", local_model_path)
        else:
            self.transformer = Model(config=cfg)

    def forward(self, tokens):
        return self.transformer(**tokens, return_dict=True)

    def add_adapter(self, name, config):
        self.transformer.add_adapter(name, config)


