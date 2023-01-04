# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

class SingleModel(object):

    def __init__(self, checkpoint_path):
        if checkpoint_path in ["npm", "npm-single"]:
            checkpoint_path = "facebook/" + checkpoint_path

        is_registered = checkpoint_path.startswith("roberta-") or checkpoint_path.startswith("facebook/")

        if is_registered:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
            print ("Loaded from HF Hub:", checkpoint_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/npm")
            # loading from trained checkpoint
            state_dict = torch.load(checkpoint_path)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            encoder_state_dict = {".".join(k.split(".")[2:]): v for k, v in state_dict.items()}
            self.model = AutoModelForMaskedLM.from_pretrained("facebook/npm", state_dict=encoder_state_dict)
            print ("Loaded from a local checkpoint:", checkpoint_path)

        self.model.cuda()
        self.model.eval()
        self.cnt = 0

    def forward(self, input_ids, idx):
        if self.cnt < 3:
            print (self.tokenizer.decode(input_ids[0]))

        outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
        logits = outputs.logits[0, idx, :]
        query = outputs["hidden_states"][-1][:, idx, :]
        self.cnt += 1

        return logits, query

class Model(SingleModel):

    def forward(self, input_ids, idx):
        assert len(input_ids)==1
        assert input_ids[0, idx]==self.tokenizer.mask_token_id
        new_input_ids = torch.cat([input_ids[:, :idx+1], input_ids[:, idx:]], -1)
        assert len(new_input_ids[0])==len(input_ids[0])+1
        assert new_input_ids[0, idx]==new_input_ids[0, idx+1]==self.tokenizer.mask_token_id

        if self.cnt < 3:
            print (self.tokenizer.decode(new_input_ids[0]))

        outputs = self.model(new_input_ids, output_hidden_states=True, return_dict=True)
        logits = outputs.logits[0, idx, :]
        start_query = outputs["hidden_states"][-1][:, idx, :]
        end_query = outputs["hidden_states"][-1][:, idx+1, :]
        self.cnt += 1

        return logits, (start_query, end_query)


