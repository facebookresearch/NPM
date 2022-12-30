# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import time
import numpy as np
import os
import pickle as pkl
import datetime
import re
import string

from collections import defaultdict, Counter
from scipy.special import softmax, log_softmax

from npm.npm_single import NPMSingle
from task.utils_eval import normalize_answer

import torch.nn.functional as F

class NPM(NPMSingle):
    def get_all_scores(self, queries):
        start_query, end_query = queries
        start_scores, start_indices, start_knn_ids = super().get_all_scores(start_query)
        end_scores, end_indices, end_knn_ids = super().get_all_scores(end_query)

        all_indices = np.concatenate([start_indices, end_indices], -1)
        knn_ids = [start_knn_ids[0] + end_knn_ids[0]]
        all_scores = np.concatenate([start_scores, end_scores], -1)
        all_scores /= self.temperature
        assert len(knn_ids)==len(all_scores)==1 and len(knn_ids[0])==len(all_scores[0])

        return all_scores, all_indices, knn_ids

    def predict_span(self, query_text, ngram_max, valid_func=None,
                     alphas=[0.0], is_question=False):

        t0 = time.time()

        # first, obtain query emb
        inputs = self.model.tokenizer(query_text)
        input_ids = inputs["input_ids"]
        assert self.model.tokenizer.mask_token_id in input_ids
        idx = input_ids.index(self.model.tokenizer.mask_token_id)
        with torch.no_grad():
            input_tensor = torch.LongTensor([input_ids]).cuda()
        _, (start_query_tensor, end_query_tensor) = self.model.forward(input_tensor, idx)
        start_query = start_query_tensor.detach().cpu().numpy()
        end_query = end_query_tensor.detach().cpu().numpy()

        pos2ngram = {}
        predictions = {}

        # this is a utility function that finds all possible spans
        # composed with the top k start indices and end indices
        def get_candidates(start_indices, end_indices):
            consider_string_boundary = self.dstore.consider_string_boundary

            start_triples = self.dstore._get_token_position(start_indices.tolist(),
                                                            ngram_after=ngram_max)
            end_triples = self.dstore._get_token_position(end_indices.tolist(),
                                                        ngram_before=ngram_max)

            all_start_indices = set()
            all_end_indices = set()
            all_start_and_end = set()

            for (block_idx, token_indices, vocabs), start_token_idx in zip(start_triples[0], start_indices[0]):

                if consider_string_boundary and token_indices[0] not in self.dstore.orig_block_idx_to_valid_start[block_idx]:
                    continue
                all_start_indices.add(start_token_idx)
                end_token_idx = start_token_idx

                for j in range(len(token_indices)):

                    is_valid_start = token_indices[j] in self.dstore.orig_block_idx_to_valid_start[block_idx]
                    is_valid_end = token_indices[j] in self.dstore.orig_block_idx_to_valid_end[block_idx]

                    if self.dstore.embs_consider_boundary and not (is_valid_start or is_valid_end):
                        continue

                    if (not consider_string_boundary) or is_valid_end:
                        ngram = vocabs[:j+1]
                        ngram_pos = (start_token_idx, end_token_idx)
                        # ngram_pos = (block_idx, token_indices[0], token_indices[0]+j)
                        # assert len(ngram)==ngram_pos[1][1]-ngram_pos[1][0]+1
                        if valid_func is None or valid_func(ngram):
                            if ngram_pos in pos2ngram:
                                assert pos2ngram[ngram_pos]==ngram
                            else:
                                pos2ngram[ngram_pos] = ngram
                            all_end_indices.add(end_token_idx)
                            all_start_and_end.add(ngram_pos)

                    end_token_idx += 1

            for (block_idx, token_indices, vocabs), end_token_idx in zip(end_triples[0], end_indices[0]):

                if consider_string_boundary and token_indices[-1] not in self.dstore.orig_block_idx_to_valid_end[block_idx]:
                    continue
                all_end_indices.add(end_token_idx)
                start_token_idx = end_token_idx

                for j in range(len(token_indices)):

                    is_valid_start = token_indices[-j-1] in self.dstore.orig_block_idx_to_valid_start[block_idx]
                    is_valid_end = token_indices[-j-1] in self.dstore.orig_block_idx_to_valid_end[block_idx]

                    if self.dstore.embs_consider_boundary and not (is_valid_start or is_valid_end):
                        continue

                    if (not consider_string_boundary) or is_valid_start:
                        ngram = vocabs[-j-1:]
                        ngram_pos = (start_token_idx, end_token_idx)
                        # ngram_pos = (block_idx, token_indices[-1]-j, token_indices[-1])
                        # assert len(ngram)==ngram_pos[1][1]-ngram_pos[1][0]+1
                        if valid_func is None or valid_func(ngram):
                            if ngram_pos in pos2ngram:
                                assert pos2ngram[ngram_pos]==ngram
                            else:
                                pos2ngram[ngram_pos] = ngram
                            all_start_indices.add(start_token_idx)
                            all_start_and_end.add(ngram_pos)

                    start_token_idx -= 1

            return all_start_indices, all_end_indices, all_start_and_end

        def get_scores(start_indices, end_indices):
            x = self.dstore.get_embs(start_indices)
            x = torch.from_numpy(x).cuda()
            start_scores = self.get_scores(start_query_tensor, x)[0]
            start_scores = start_scores.detach().cpu().numpy()

            x = self.dstore.get_embs(end_indices)
            x = torch.from_numpy(x).cuda()
            end_scores = self.get_scores(end_query_tensor, x)[0]
            end_scores = end_scores.detach().cpu().numpy()

            return start_scores, end_scores

        t1 = time.time()

        # main code starts from here
        if self.dstore.restricted:
            # find passaages to restricted
            if query_text in self.dstore.restricted_dict:
                block_ids = self.dstore.restricted_dict[query_text]
            else:
                block_ids = self.dstore.searcher.search(query_text, is_question=is_question)
                self.dstore.restricted_dict[query_text] = block_ids

            #valid_idxs = [v for block_id in block_ids
            #              for v in self.dstore.orig_block_idx_to_emb_token_idx[block_id]]
            valid_idxs = []
            for block_id in block_ids:
                start, end = self.dstore.orig_block_idx_to_emb_token_idx[block_id:block_id+2]
                valid_idxs += list(range(start, end))
            start_indices = np.array([valid_idxs])
            end_indices = np.array([valid_idxs])

        else:
            _, start_indices = self.dstore.search(start_query, k=self.k)
            _, end_indices = self.dstore.search(end_query, k=self.k)

        if start_indices.shape[1]==end_indices.shape[1]==0:
            for alpha in alphas:
                predictions["a={}".format(alpha)] = None
            return predictions

        t2 = time.time()

        if self.dstore.restricted:
            start_scores, end_scores = get_scores(start_indices, end_indices)
            _, _, all_start_and_end = get_candidates(start_indices, end_indices)

            all_start_indices = start_indices[0].tolist()
            all_end_indices = end_indices[0].tolist()
            all_start_scores = start_scores
            all_end_scores = end_scores

        else:
            all_start_indices, all_end_indices, all_start_and_end = get_candidates(start_indices, end_indices)

            all_start_indices = sorted(all_start_indices)
            all_end_indices = sorted(all_end_indices)

            all_start_scores, all_end_scores = get_scores(all_start_indices, all_end_indices)

        all_start_scores = softmax(all_start_scores / self.temperature, -1)
        all_end_scores = softmax(all_end_scores / self.temperature, -1)

        idx2start_score = {start_token_idx: score for start_token_idx, score
                           in zip(all_start_indices, all_start_scores)}
        idx2end_score = {end_token_idx: score for end_token_idx, score
                         in zip(all_end_indices, all_end_scores)}

        pos2score = {}
        ngram2score = defaultdict(list)

        t3 = time.time()

        # now, assign scores to possible ngrams
        for (start, end) in all_start_and_end:
            try:
                assert start in idx2start_score
                assert end in idx2end_score
            except Exception:
                from IPython import embed; embed(); exit()
            score = idx2start_score[start] + idx2end_score[end]

            pos2score[(start, end)] = score
            ngram2score[tuple(pos2ngram[(start, end)])].append(score)

        if len(pos2score)==len(ngram2score)==0:
            for alpha in alphas:
                predictions["a={}".format(alpha)] = None
            return predictions

        assert len(pos2score)>0 and len(ngram2score)>0

        t4 = time.time()

        for alpha in alphas:
            def key_func(x, alpha=alpha):
                return -np.sum(x[1]) * np.power(len(x[0]), alpha)

            top1_ngram_score_pair = min(ngram2score.items(), key=key_func)
            top1_ngram = list(top1_ngram_score_pair[0])

            predictions["a={}".format(alpha)] = top1_ngram

        t5 = time.time()

        return predictions

    def get_query(self, input_text):
        inputs = self.model.tokenizer(input_text)
        input_ids = inputs["input_ids"]
        assert self.model.tokenizer.mask_token_id in input_ids
        idx = input_ids.index(self.model.tokenizer.mask_token_id)
        with torch.no_grad():
            input_tensor = torch.LongTensor([input_ids]).cuda()
        _, query = self.model.forward(input_tensor, idx)
        return query

    def evaluate_open(self, task):
        all_predictions = []
        mask = self.get_stopword_mask()
        do_restricted = self.dstore is not None and self.dstore.restricted is not None
        def valid_func(tokens):
            return np.sum(mask[tokens])==0
        if "translation" in str(task):
            alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            ngram_max = 20
        else:
            alphas = [0.0, 0.5, 1.0]
            ngram_max = 10

        for ex in tqdm(task.examples):
            dic = self.predict_span(
                ex["input"],
                ngram_max=ngram_max,
                valid_func=valid_func,
                alphas=alphas,
                is_question=task.is_question,
            )
            dic = {k: '' if v is None else self.decode(v) for k, v in dic.items()}
            all_predictions.append(dic)

        # compute accuracy
        references = [[normalize_answer(answer) for answer in ex["answers"]] for ex in task.examples]
        for k in all_predictions[0]:
            predictions = [normalize_answer(p[k]) for p in all_predictions]
            accs = [prediction in reference for prediction, reference in zip(predictions, references)]

            if task.ngrams is not None:
                accs_dict = defaultdict(list)
                for acc, ngram in zip(accs, task.ngrams):
                    accs_dict[ngram].append(acc)
                acc = np.mean([np.mean(v) for k, v in accs_dict.items()])
                print ("\t%s\tMacro EM=%.1f%%" % (k, 100*acc))
            else:
                acc = np.mean(accs)
                print ("\t%s\tEM=%.1f%%" % (k, 100*acc))

        return all_predictions


