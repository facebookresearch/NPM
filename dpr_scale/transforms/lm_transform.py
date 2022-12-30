# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import numpy as np
import hydra
import torch
import math
import torch.nn as nn
import json

class LanguageModelingTransform(nn.Module):
    def __init__(
        self,
        bidirectional=False,
        masking=None,
        masking_ratio=0.0,
        preprocessed_tokenizer_type="roberta",
        exactly_follow_roberta=False,
        enforce_masking_positives=False
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.masking = masking
        self.masking_ratio = masking_ratio
        self.preprocessed_tokenizer_type = preprocessed_tokenizer_type
        assert self.preprocessed_tokenizer_type in ["gpt", "roberta"]

        self.enforce_masking_positives = enforce_masking_positives
        if enforce_masking_positives:
            assert masking and masking_ratio > 0

        # if True, 80% mask, 10% original, 10% replaced
        self.exactly_follow_roberta = exactly_follow_roberta

        if self.bidirectional:
            from transformers import RobertaTokenizer
            roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

            self.tokenizer = roberta_tokenizer
            self.mask_id = roberta_tokenizer.mask_token_id
            self.bos_id = roberta_tokenizer.bos_token_id
            self.eos_id = roberta_tokenizer.eos_token_id
            self.pad_id = roberta_tokenizer.pad_token_id

            if masking is not None:
                assert masking in ["uniform", "span", "span-merge", "span-merge-two"]
                assert masking_ratio > 0
            else:
                assert masking_ratio == 0

        else:
            assert self.preprocessed_tokenizer_type=="gpt"
            assert masking is None and masking_ratio==0.0

    def get_mask_id(self, token):
        if self.exactly_follow_roberta:
            p = np.random.random()
            if p<0.1:
                return token
            if p<0.2:
                return np.random.choice(range(3, len(self.tokenizer)-1))
        return self.mask_id

    def transform_to_roberta(self, tokens, attention_mask,
                             add_bos_id=False,
                             add_eos_id=False):

        assert type(tokens)==list and type(attention_mask)==list
        assert len(tokens)==len(attention_mask)
        new_input_ids, new_attention_mask = [], []
        masked_input_ids = [] # input_ids with mask
        label_mask = [] # indicator on which tokens are those to be predicted

        if type(tokens[0])==list and type(attention_mask[0])==list:
            for token, mask in zip(tokens, attention_mask):
                _input_ids, _attention_mask, _masked_input_ids, _label_mask = self.transform_to_roberta(token, mask)
                new_input_ids.append(_input_ids)
                new_attention_mask.append(_attention_mask)
                masked_input_ids.append(_masked_input_ids)
                label_mask.append(_label_mask)

        elif self.preprocessed_tokenizer_type=="roberta":
            new_input_ids = tokens
            new_attention_mask = attention_mask

            # np.random.seed(2022) # for debugging

            for i, (token, mask) in enumerate(zip(tokens, attention_mask)):
                assert type(token)==type(mask)==int
                if mask and token not in [self.bos_id, self.eos_id] and \
                        self.masking=="uniform" and np.random.random()<self.masking_ratio:
                    masked_input_ids.append(self.get_mask_id(token))
                    label_mask.append(1)
                else:
                    masked_input_ids.append(token)
                    label_mask.append(0)
        else:
            raise NotImplementedError()

        return new_input_ids, new_attention_mask, masked_input_ids, label_mask

    def forward(self, batch, stage="train"):
        #print ("transform forward called")
        input_ids = []
        attention_mask = []
        masked_input_ids = []
        masked_attention_mask = []
        labels = []

        rows = batch if type(batch) is list else batch[self.text_column]
        for row in rows:
            dp = json.loads(row)
            input_ids.append(dp["input_ids"])
            attention_mask.append(dp["attention_mask"])

            if self.masking in ["span-merge", "span-merge-two"]:
                masked_input_ids.append(dp["merged_masked_input_ids"])
                masked_attention_mask.append(dp["merged_attention_mask"])
                labels.append(dp["merged_labels"])

            elif self.masking=="span":
                masked_input_ids.append(dp["masked_input_ids"])

        # span masking with each span represented with two masks
        if self.masking=="span-merge":
            assert len(labels)==1, len(labels)
            flatten_labels = [l for _labels in labels[0] for l in _labels]
            L = np.max([len(l) for l in flatten_labels])
            for i, fl in enumerate(flatten_labels):
                flatten_labels[i] = fl + [0]*(L-len(fl))

            '''there was a minor bug in merged_attention_mask, so will fix it'''
            assert len(masked_input_ids)==len(masked_attention_mask)==1
            for i, (curr_input_ids, curr_attention_mask) in enumerate(zip(masked_input_ids[0], masked_attention_mask[0])):
                curr_input_ids = masked_input_ids[0][i]
                curr_attention_mask = masked_attention_mask[0][i]
                # if there is attention_mask==0 in the middle, convert it to 1
                # if there is attention_mask==0 while input_ids!=0, convert it to 1
                for j in range(len(curr_input_ids)-1, -1, -1):
                    curr_input_id = curr_input_ids[j]
                    curr_mask = curr_attention_mask[j]
                    if curr_mask>0:
                        continue
                    if curr_input_id>0 or np.sum(curr_attention_mask[j+1:])>0:
                        masked_attention_mask[0][i][j] = 1

                if 0 in masked_attention_mask[0][i]:
                    L = masked_attention_mask[0][i].index(0)
                    assert np.all(masked_attention_mask[0][i][:L])
                    assert not np.any(masked_attention_mask[0][i][L:])
                    assert not np.any(masked_input_ids[0][i][L:])
                else:
                    assert np.all(masked_attention_mask[0][i])

            assert len(masked_input_ids)==len(masked_attention_mask)==1
            masked_input_ids = masked_input_ids[0]
            masked_attention_mask = masked_attention_mask[0]

            _masked_input_ids, _masked_attention_mask = [], []
            L = len(masked_input_ids[0])
            for curr_input_ids, curr_attention_mask in zip(masked_input_ids, masked_attention_mask):
                offset = 0
                mask_cnt = curr_input_ids.count(self.mask_id)
                while offset < len(curr_input_ids) and self.mask_id in curr_input_ids[offset:]:
                    idx = offset + curr_input_ids[offset:].index(self.mask_id)
                    curr_input_ids = curr_input_ids[:idx] + [self.mask_id, self.mask_id] + curr_input_ids[idx+1:]
                    offset = idx+2
                curr_attention_mask = [1] * mask_cnt + curr_attention_mask
                assert mask_cnt*2==curr_input_ids.count(self.mask_id)
                assert len(curr_input_ids)==len(curr_attention_mask)==L+mask_cnt

                if 0 in curr_attention_mask:
                    l = curr_attention_mask.index(0)
                    curr_input_ids = curr_input_ids[:l]
                    curr_attention_mask = curr_attention_mask[:l]

                _masked_input_ids.append(curr_input_ids)
                _masked_attention_mask.append(curr_attention_mask)

            L = max(
                np.max([len(_input_ids) for _input_ids in _masked_input_ids]),
                np.max([np.sum(mask) for mask in attention_mask[0]]))
            assert L<=320 #384
            L = 320 #384

            input_ids = [
                _input_ids[:min(L, np.sum(_attention_mask))] + [0]*max(0, L-np.sum(_attention_mask))
                for _input_ids, _attention_mask in zip(input_ids[0], attention_mask[0])
            ]
            attention_mask = [
                _attention_mask[:min(L, np.sum(_attention_mask))] + [0]*max(0, L-np.sum(_attention_mask))
                for _attention_mask in attention_mask[0]
            ]
            masked_input_ids = [
                _input_ids[:L] + [0]*max(0, L-len(_input_ids))
                for _input_ids in _masked_input_ids
            ]
            masked_attention_mask = [
                _attention_mask[:L] + [0]*max(0, L-len(_attention_mask))
                for _attention_mask in _masked_attention_mask
            ]

            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            masked_input_ids = torch.LongTensor(masked_input_ids)
            masked_attention_mask = torch.LongTensor(masked_attention_mask)
            labels = torch.LongTensor(flatten_labels)
            label_mask = masked_input_ids==self.mask_id

            assert len(flatten_labels)*2==torch.sum(label_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "masked_input_ids": masked_input_ids,
                "masked_attention_mask": masked_attention_mask,
                "labels": labels,
                "label_mask": label_mask
            }

        # span masking with each token in the span is represented with a mask
        elif self.masking=="span":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "masked_input_ids": torch.LongTensor(masked_input_ids),
                "attention_mask": torch.LongTensor(attention_mask),
            }

        elif self.bidirectional:
            if self.enforce_masking_positives:
                assert self.masking=="uniform"

                input_ids = torch.LongTensor(input_ids)
                attention_mask = torch.LongTensor(attention_mask)
                assert input_ids.shape[0]==1
                BS = input_ids.shape[1]
                length = input_ids.shape[2]
                labels = torch.logical_and(
                    input_ids.reshape(-1).unsqueeze(-1)==input_ids.reshape(-1).unsqueeze(0),
                    attention_mask.reshape(-1).unsqueeze(-1)*attention_mask.reshape(-1).unsqueeze(0)
                )
                maskout = torch.block_diag(*torch.ones((BS, length, length), dtype=torch.bool))
                labels = torch.logical_and(labels, ~maskout)

                labels = torch.any(labels, -1).reshape(1, BS, length)
                labels = torch.logical_and(
                    labels,
                    torch.logical_and(
                        input_ids!=self.bos_id, input_ids!=self.eos_id))

                masked_input_ids = input_ids.clone()
                label_mask = torch.zeros_like(masked_input_ids)
                masking_ratio = torch.sum(attention_mask) * self.masking_ratio / torch.sum(labels)

                for i in range(BS):
                    for j in range(length):
                        if labels[0, i, j] and np.random.random() < masking_ratio:
                            masked_input_ids[0, i, j] = self.mask_id
                            label_mask[0, i, j] = 1

            else:
                input_ids, attention_mask, masked_input_ids, label_mask = self.transform_to_roberta(input_ids, attention_mask)
                input_ids = torch.LongTensor(input_ids)
                masked_input_ids = torch.LongTensor(masked_input_ids)
                attention_mask = torch.LongTensor(attention_mask)
                label_mask = torch.LongTensor(label_mask)
                # print (torch.mean(label_mask.float()))

            return {
                "input_ids": input_ids,
                "masked_input_ids": masked_input_ids,
                "attention_mask": attention_mask,
                "label_mask": label_mask,
            }

        else:
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attention_mask": torch.LongTensor(attention_mask),
            }

