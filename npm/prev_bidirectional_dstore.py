# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import numpy as np
import json
import torch
import faiss
import time
import glob

from tqdm import tqdm
from scipy.special import log_softmax

from dstore import DataStore

class BidirectionalDataStore(DataStore):
    def load_stopwords(self):
        if self.remove_stopwords:
            stopwords = set()
            stopwords_dir = "/private/home/sewonmin/token-retrieval/task_data"
            with open(os.path.join(stopwords_dir, "roberta_stopwords.txt")) as f:
                for line in f:
                    stopwords.add(int(line.strip()))
        else:
            stopwords = None
        return stopwords

    def parse_line(self, line, stopwords, save_dstore_size_only=False, return_dstore_size_only=False):
        dp = json.loads(line)
        input_ids = dp["input_ids"][:np.sum(dp["attention_mask"])]

        # Now compute `dstore_size`, `token_idx_to_block_idx`,
        # `doc_idx_to_block_idx`, (doesn't really matter for prompting)
        # and `token_counter`

        dstore_size = 0

        for i, curr_token in enumerate(input_ids):
            if curr_token in [0, 2]:
                continue
            if self.remove_stopwords and curr_token in stopwords:
                continue
            if return_dstore_size_only:
                dstore_size += 1
                continue
            self.token_idx_to_block_idx[len(self.token_idx_to_block_idx)] = (len(self.blocks), i)
            if self.token_counter is not None:
                self.token_counter[curr_token] += 1
            dstore_size += 1

            doc_idx = dp["doc_idx"] if "doc_idx" in dp else -1
            self.block_idx_to_doc_idx[len(self.blocks)] = doc_idx
            self.doc_idx_to_block_idx[doc_idx] = len(self.blocks)

        if return_dstore_size_only:
            return dstore_size

        if save_dstore_size_only:
            self.blocks.append({})
        else:
            raw_text = dp["raw_text"] if "raw_text" in dp else dp["contents"]
            self.blocks.append({"input_ids": input_ids,
                                "raw_text": raw_text})

        return dstore_size

    def _get_token(self, token_idx, ngram=1):
        block_i, token_i = self.token_idx_to_block_idx[token_idx]
        input_ids = self.blocks[block_i]["input_ids"]
        assert token_i < len(input_ids)
        token_i = input_ids[token_i:token_i+ngram]
        return block_i, token_i

    def _get_token_position(self, token_idx, ngram_before=1, ngram_after=1):
        if type(token_idx)==list:
            return [self._get_token_position(_token_idx,
                                             ngram_before=ngram_before,
                                             ngram_after=ngram_after)
                    for _token_idx in token_idx]
        block_i, token_i = self.token_idx_to_block_idx[token_idx]
        input_ids = self.blocks[block_i]["input_ids"]
        assert token_i < len(input_ids)

        if ngram_before==1:
            # just take ngram after this
            token_range = [token_i]
            for j in range(token_i+1, min(len(input_ids), token_i+ngram_after)):
                if input_ids[j] in [0, 2]:
                    break
                token_range.append(j)
        elif ngram_after==1:
            token_range = [token_i]
            for j in range(token_i-1, max(0, token_i-ngram_before+1), -1):
                if input_ids[j] in [0, 2]:
                    break
                token_range = [j] + token_range
        else:
            raise NotImplementedError()

        assert np.all([i+1==j for i, j in zip(token_range, token_range[1:])])
        return block_i, token_range, [input_ids[i] for i in token_range]

class BidirectionalDataStoreUnion(BidirectionalDataStore):
    def __init__(self, settings,
                 data_path=None, model_dir=None,
                 do_load_data=True, do_load_embeds=True, do_load_index=True,
                 dstore_distance_type="l2", exact=False, dimension=1280, probe=8,
                 remove_stopwords=False, tokenizer_type=None):
        self.dstores = []
        for setting in settings:
            self.dstores.append(BidirectionalDataStore(setting=setting,
                                          data_path=data_path,
                                          model_dir=model_dir,
                                          do_load_data=do_load_data,
                                          do_load_embeds=do_load_embeds,
                                          do_load_index=do_load_index,
                                          dstore_distance_type=dstore_distance_type,
                                          exact=exact,
                                          dimension=dimension,
                                          probe=probe,
                                          remove_stopwords=remove_stopwords,
                                                       tokenizer_type=tokenizer_type))
        self.dimension = self.dstores[0].dimension

    def get_frequency(self, tokens):
        return [np.sum([dstore.token_counter.get(token, 0) for dstore in self.dstores])
                for token in tokens]

    def search(self, queries, k):
        all_scores, all_indices = [], []
        for dstore_idx, dstore in enumerate(self.dstores):
            scores, indices = dstore.search(queries, k)
            all_scores.append(scores)
            all_indices.append(indices)

        return np.stack(all_scores, 0), np.stack(all_indices, 0)

    def get_block_idx_and_token(self, indices, token_only, ngram):
        assert token_only # not implemented otherwise
        assert len(indices)==len(self.dstores)
        tokens = None
        for dstore, _indices in zip(self.dstores, indices):
            curr_tokens = dstore.get_block_idx_and_token(_indices, token_only=True, ngram=ngram)
            if tokens is None:
                tokens = curr_tokens.copy()
            else:
                for i, (_tokens, _curr_tokens) in enumerate(zip(tokens, curr_tokens)):
                    tokens[i] += _curr_tokens
        return tokens

    def get_embs(self, indices):
        assert len(indices)==len(self.dstores)
        embs = []
        for dstore, _indices in zip(self.dstores, indices):
            embs.append(np.array(dstore.get_embs(_indices).tolist()))
        return np.concatenate(embs, -2)

    def get_context(self, indices, decode_func):
        raise NotImplementedError()

    def get_frequency(self, tokens):
        return np.sum([ds.token_counter.get(token, 0) for token in tokens for ds in self.dstores])



