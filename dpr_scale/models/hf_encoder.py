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
from transformers import AutoModelForMaskedLM, AutoConfig

class Encoder(nn.Module):
    def __init__(
        self,
        model_path: str = "roberta-large",
        initialize: bool = True,
        dropout: float = 0.1,
        num_hidden_layers = None,
        hidden_size = None,
        vocab_size = None,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)

        cfg = AutoConfig.from_pretrained(local_model_path)
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout
        if num_hidden_layers is not None:
            cfg.num_hidden_layers = num_hidden_layers
        if hidden_size is not None:
            cfg.hidden_size = hidden_size
        if vocab_size is not None:
            cfg.vocab_size = vocab_size

        if initialize:
            self.transformer = AutoModelForMaskedLM.from_pretrained(local_model_path, config=cfg)
            print ("Initializing from", local_model_path)
        else:
            self.transformer = AutoModelForMaskedLM.from_pretrained(config=cfg)

    def forward(self, tokens):
        return self.transformer(**tokens, return_dict=True)

    def add_adapter(self, name, config):
        self.transformer.add_adapter(name, config)


