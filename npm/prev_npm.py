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

import torch.nn.functional as F

class NPM(object):
    def __init__(self, model, dstore, k):
        self.model = model
        self.k = k

        self.n_vocab = len(self.model.tokenizer)

        if self.scoring.endswith("_gen"):
            self.label2synonym = None
            self.label2synonym_id = None

            self.label2word = [self.encoder._decode(idx) for idx in range(len(self.encoder))]
            self.word2label = {v:k for k, v in enumerate(self.label2word)}
            self.label2word_id = np.arange(len(self.encoder))

            self.cache_valid_candidates = {}

        else:
            if "label2synonym" in examples[0]:
                self.label2synonym = examples[0]["label2synonym"]
                self.label2synonym_id = self.init_label2word_id(self.label2synonym)
                assert np.all([v.shape[-1]==1 for v in self.label2synonym_id.values()])

                if dstore is not None:
                    self.frequency_dict = {}
                    n_tokens = []
                    for label, tokens in self.label2synonym_id.items():
                        tokens = tokens.detach().cpu().tolist()
                        tokens = [token[0] for token in tokens]
                        freq = dstore.get_frequency(tokens)
                        self.frequency_dict[label] = np.sum(freq)

            else:
                self.label2synonym = None
                self.label2synonym_id = None

            self.labels = examples[0]["label_list"]
            self.labels_id = self.encoder(self.labels)["input_ids"]

            self.label2word = {i: [v] for i, v in enumerate(examples[0]["label_list"])}
            self.label2word_id = self.init_label2word_id(self.label2word)
            # print (self.label2word)

        self.dstore = dstore
        self.dataset = dataset

    def decode(self, ids):
        return self.encoder.decode(ids)

    def get_scores(self, queries, x):
        if type(queries)==np.ndarray:
            all_scores = np.inner(queries, x).squeeze(1) / np.sqrt(self.dstore.dimension)
        else:
            all_scores = torch.inner(queries, x).squeeze(1) / np.sqrt(self.dstore.dimension)
        return all_scores

    def get_knn_scores(self,
                       queries,
                       return_context=False,
                       extra_contexts=None,
                       ngram=1):

        def _do_search(queries):
            all_scores, all_indices = self.dstore.search(
                queries, #.detach().cpu().numpy(),
                k=self.k)
            knn_ids = self.dstore.get_block_idx_and_token(
                all_indices.tolist(), token_only=True, ngram=1)
            if ngram==1:
                def _squeeze(values):
                    if values=="EOS":
                        return values
                    assert type(values)==list, values
                    if len(values)==1 and type(values[0])!=list:
                        return values[0]
                    return [_squeeze(v) for v in values]
                knn_ids = _squeeze(knn_ids)
            return all_indices, knn_ids, all_scores
        if type(queries)==tuple and not self.use_cache:
            start_query, end_query = queries

            def _get_scores_from_query(query):
                query = query.detach().cpu().numpy()
                all_indices, knn_ids, all_scores = _do_search(query)
                if self.use_faiss_scores:
                    all_scores = -all_scores
                else:
                    x = self.dstore.get_embs(all_indices)
                    #x = torch.from_numpy(x).cuda()
                    all_scores = self.get_scores(query, x)
                    #all_scores = all_scores.detach().cpu().numpy()
                return all_indices, knn_ids, all_scores

            start_indices, start_knn_ids, start_scores = _get_scores_from_query(start_query)
            end_indices, end_knn_ids, end_scores = _get_scores_from_query(end_query)

            all_indices = np.concatenate([start_indices, end_indices], -1)
            knn_ids = [start_knn_ids[0] + end_knn_ids[0]]
            all_scores = np.concatenate([start_scores, end_scores], -1)

            assert len(knn_ids)==len(all_scores)==1 and len(knn_ids[0])==len(all_scores[0])

            if self.save_cache:
                self.cache.append([all_indices, knn_ids, all_scores])

        elif self.use_cache:
            all_indices, knn_ids, all_scores = self.cache[self.cache_idx]
            self.cache_idx += 1

        else:
            queries = queries.detach().cpu().numpy()
            all_indices, knn_ids, all_scores = _do_search(queries)
            if self.use_faiss_scores:
                all_scores = -all_scores
            else:
                x = self.dstore.get_embs(all_indices)
                all_scores = self.get_scores(queries, x)
            if self.save_cache:
                self.cache.append([all_indices, knn_ids, all_scores])

        if return_context:
            sorted_all_indices = all_indices[0, np.argsort(-all_scores[0])]
        # need to manually exclude EOS tokens
        excluded = set([i for i, knn_id in enumerate(knn_ids[0])
                        if knn_id in ["EOS"]]) #, 198, 1849]])
        if len(excluded)>0:
            included = [i for i in range(len(knn_ids[0])) if i not in excluded]
            knn_ids = [[knn_ids[0][i] for i in included]]
            all_scores = all_scores[:, np.array(included)]
            if return_context:
                sorted_all_indices = sorted_all_indices[np.array(included)]

        if return_context:
            assert len(sorted_all_indices)==len(knn_ids[0])

        if extra_contexts is not None:
            extra_embs, idx_to_vocab, idx_to_context = self.add_context(extra_contexts)
            extra_scores = self.get_scores(queries, extra_embs)
            all_scores = np.concatenate([all_scores, extra_scores], -1)
            knn_ids[0] += idx_to_vocab

        k = all_scores.shape[1]
        assert len(knn_ids)==len(all_scores)==1 and \
            len(knn_ids[0])==len(all_scores[0])

        probs = softmax(all_scores / self.temperature, -1)
        assert len(knn_ids)==1 and probs.shape[0]==1 and len(knn_ids[0])==len(probs[0])
        full_knn_scores = {}
        for vocab, p in zip(knn_ids[0], probs[0]):
            if vocab not in full_knn_scores:
                full_knn_scores[vocab] = 0
            full_knn_scores[vocab] += p

        if self.scoring.startswith("logsoftmax_"):
            full_knn_scores = {v: np.log(p) for v, p in full_knn_scores.items()}
        elif self.scoring.startswith("softmax_"):
            pass
        else:
            raise NotImplementedError()

        if self.scoring.startswith("softmax"):
            label2knn_prob = np.zeros((self.n_vocab, ))
        else:
            label2knn_prob = -1e10 * np.ones((self.n_vocab, ))

        for vocab, p in full_knn_scores.items():
            label2knn_prob[vocab] = p

        if return_context:
            def decode_func(input_ids, token_i):
                assert token_i < len(input_ids)
                if token_i==len(input_ids)-1:
                    return self.encoder.decode(input_ids) + " " + colored("EOS", "red"), "EOS"
                retrieved_token = self.encoder.decode([input_ids[token_i+1]])
                return self.encoder.decode(input_ids[:token_i+1]) + \
                    colored(retrieved_token, "red") + \
                    self.encoder.decode(input_ids[token_i+2:]), retrieved_token

            context = self.dstore.get_context(sorted_all_indices.tolist(), decode_func)
            if extra_contexts is not None:
                context += [decode_func(input_ids, token_i) for input_ids, token_i in idx_to_context]
            assert len(context)==len(all_scores[0])
            sorted_context_and_scores = sorted(zip(context, all_scores[0]),
                                               key=lambda x: -x[1])

            return label2knn_prob, sorted_context_and_scores

        return label2knn_prob

    def compute_LM_prob4tokens(self, logits):
        logits = logits.detach().cpu()
        if self.scoring.startswith("logsoftmax_"):
            last_token_softmax = torch.log_softmax(logits, dim=-1)
        elif self.scoring.startswith("softmax_"):
            last_token_softmax = torch.softmax(logits, dim=-1)
        return last_token_softmax.numpy()

    def predict(self,
                input_text,
                knn_input_text,
                label_list,
                label2id=None,
                lm_synonym=False,
                knn_synonym=True,
                return_context=False):
        label_prob_dict = self.eval_one_ex(input_text,
                                               knn_input_text,
                                               return_context=return_context)
        if return_context:
            label_prob_dict, retrieved_context = label_prob_dict

        lm_prob, knn_prob = label_prob_dict

        def convert_to_label_prob(prob, label2id=None, synonym=False):
            if label2id is None:
                label2id = self.label2synonym_id if synonym and self.label2synonym_id is not None else self.label2word_id
                label2id = {label: ids.detach().cpu().numpy() for label, ids in label2id.items()}

            prob = np.array([np.sum(prob[label2id[label]]) for label in range(len(label2id))])

            if self.scoring.endswith("_vocab"):
                pass
            elif self.scoring=="logsoftmax_label":
                prob = np.exp(prob)
                if np.sum(prob)==0:
                    prob = np.zeros_like(prob)
                else:
                    prob /= np.sum(prob)
                    prob = np.log(prob)
            elif self.scoring=="softmax_label":
                if np.sum(prob)==0:
                    prob = np.zeros_like(prob)
                else:
                    prob /= np.sum(prob)
            else:
                raise NotImplementedError()
            return prob

        _lm_prob = convert_to_label_prob(lm_prob, label2id=label2id, synonym=lm_synonym)
        _knn_prob = convert_to_label_prob(knn_prob, label2id=label2id, synonym=knn_synonym)

        all_pred = {}
        for knn_lambda in self.knn_lambdas:
            prob = self.interpolate(knn_lambda, _lm_prob, _knn_prob)
            all_pred[knn_lambda] = prob #np.argmax(prob)

        if return_context:
            return all_pred, retrieved_context

        return all_pred

    def evaluate(self, lm_synonym=False, knn_synonym=True, do_audit=False):
        all_predictions = []
        assert np.all([word_ids.shape[-1]==1 for word_ids in self.label2word_id.values()])

        for ex in tqdm(self.examples):
            input_text = ex["options"][0]["premise"]
            knn_input_text = ex["options"][0]["premise"] #["knn_premise"]
            label2id = None

            prediction = self.predict(
                    input_text=input_text,
                    knn_input_text=knn_input_text,
                    label_list=ex["label_list"] if "label_list" in ex else None,
                    label2id=label2id,
                    lm_synonym=lm_synonym,
                    knn_synonym=knn_synonym)
            all_predictions.append(prediction)


    def init_label2word_id(self, label2synonym):
        label2synonym_id = {}
        for k, v in label2synonym.items():
            synonym_id = []
            for word in v:
                tokens = self.encoder(word)["input_ids"]
                if len(tokens)!=1:
                    assert len(tokens)==3
                    assert (tokens[0]==0 and tokens[-1]==2) or (tokens[0]==101 and tokens[-1]==102)
                    tokens = tokens[1:-1]
                assert len(tokens)==1
                synonym_id.append(tokens)
            label2synonym_id[k] = torch.LongTensor(synonym_id).cuda()
        return label2synonym_id

    def get_stopword_mask(self, name="stopwords", stopwords=set()):
        mask = np.zeros((len(self.label2word), ))
        stopwords = set()

        if "roberta" not in self.encoder.__class__.__name__:
            from transformers import RobertaTokenizer
            roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            with open("task_data/roberta_" + name + ".txt") as f:
                for line in f:
                    stopwords.add(roberta_tokenizer._decode(int(line.strip())))
            for _id, word in enumerate(self.label2word):
                if word in stopwords:
                    mask[_id] = -1e10
        else:
            # maskout stopwords
            with open("task_data/roberta_" + name + ".txt") as f:
                for line in f:
                    stopwords.add(int(line.strip()))
            mask[np.array(list(stopwords))] = -1e10
        return mask

    def eval_one_ex(self, input_texts, knn_input_texts,
                    extra_contexts=None, return_context=False):

        assert type(input_texts)==str

        # this is T5
        if self.model.model is not None and "T5" in str(self.model.model.__class__):
            mask_id = len(self.encoder)-1
            if "<mask>" in input_texts:
                input_texts = input_texts.replace("<mask>", "<extra_id_0>")
            else:
                input_texts = input_texts + " <extra_id_0>."
            assert "<extra_id_0>" in input_texts
            input_ids = self.encoder(input_texts).input_ids
            if self.dataset=="yahoo":
                # looks like it gets OOM if input_ids is too long
                if len(input_ids)>128:
                    input_ids = input_ids[-128:]
            assert mask_id in input_ids
            input_ids=torch.LongTensor([input_ids]).cuda()
            decoder_input_ids=torch.LongTensor([[0, 32099]]).cuda()

            logits = self.model.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits[0, -1, :]
            logits = logits.detach().cpu().numpy()[:self.n_vocab]

            return logits, np.zeros_like(logits)

        if "<mask>" not in input_texts:
            input_texts += " {}.".format(self.encoder.mask_token)
        inputs = self.encoder.encode_plus(input_texts)
        input_ids = inputs["input_ids"]

        mask_id = self.encoder.mask_token_id
        idx = input_ids.index(mask_id)

        if len(input_ids) > 256:
            input_ids = input_ids[-256:]
            assert mask_id in input_ids
            idx = input_ids.index(mask_id)

        if self.use_cache:
            knn_queries = None
            label2LM_prob = np.zeros((self.n_vocab, ))
        else:
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to("cuda")

            with torch.no_grad():
                if self.model.two_mask:
                    logits, knn_queries = self.model.forward(input_ids, idx, two_mask=True)
                else:
                    logits, knn_queries = self.model.forward(input_ids, idx)

                label2LM_prob = self.compute_LM_prob4tokens(logits)

        if self.knn_lambdas!=[0.0]:
            label2knn_prob = self.get_knn_scores(knn_queries,
                                                    extra_contexts=extra_contexts,
                                                    return_context=return_context)
        else:
            label2knn_prob = np.zeros_like(label2LM_prob)

            if return_context:
                label2knn_prob, retrieved_context = label2knn_prob

        if return_context:
            return (label2LM_prob, label2knn_prob), retrieved_context

        return label2LM_prob, label2knn_prob

    def retrieve_ngram(self, queries, ngram_max, valid_func=None,
                       bm25_passages=None, oracle_passages=None):

        separate_unigram = False #True
        if self.dataset.startswith("translation"):
            alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        else:
            alphas = [0.0, 0.5, 1.0]

        pos2ngram = {}
        predictions = {}

        start_query_tensor, end_query_tensor = queries
        start_query = start_query_tensor.detach().cpu().numpy()
        end_query = end_query_tensor.detach().cpu().numpy()

        assert not self.use_cache
        #single_indices, single_scores, start_indices, start_scores, end_indices, end_scores = self.cache[self.cache_idx]
        #self.cache_idx += 1

        def get_initial_indices(mode, bm25_passages):
            assert mode in ["default", "bm25", "both"]
            if mode=="bm25":
                bm25_passages_set = set([p["offset"] for p in bm25_passages])
                valid_idxs = []
                for k, v in self.dstore.emb_token_idx_to_orig_block_idx.items():
                    if v in bm25_passages_set:
                        valid_idxs.append(k)
                start_indices = np.array([valid_idxs])
                end_indices = np.array([valid_idxs])
            else:
                _, start_indices = self.dstore.search(start_query, k=self.k)
                _, end_indices = self.dstore.search(end_query, k=self.k)

                if mode == "both":
                    start_indices = set(start_indices[0].tolist())
                    end_indices = set(end_indices[0].tolist())
                    bm25_passages_set = set([p["offset"] for p in bm25_passages])
                    for k, v in self.dstore.emb_token_idx_to_orig_block_idx.items():
                        if v in bm25_passages_set:
                            start_indices.add(v)
                            end_indices.add(v)
                    start_indices = np.array([list(start_indices)])
                    end_indices = np.array([list(end_indices)])
            return start_indices, end_indices

        def get_candidates(start_indices, end_indices):
            start_triples = self.dstore._get_token_position(start_indices.tolist(),
                                                            ngram_after=ngram_max)
            end_triples = self.dstore._get_token_position(end_indices.tolist(),
                                                        ngram_before=ngram_max)

            # cache_valid_candidates
            for block_idx, _, _ in start_triples[0] + end_triples[0]:
                if block_idx in self.cache_valid_candidates:
                    pass
                input_ids = self.dstore.blocks[block_idx]["input_ids"]
                tokens = [self.label2word[_id] for _id in input_ids]
                valid_start, valid_end = set(), set()
                for token_idx, token in enumerate(tokens):
                    if token_idx==0 or token.startswith(" "):
                        valid_start.add(token_idx)
                    elif token_idx>0 and np.any([tokens[token_idx-1].endswith(punc) for punc in string.punctuation]):
                        valid_start.add(token_idx)
                    if token_idx==len(tokens)-1 or tokens[token_idx+1].startswith(" ") or \
                            np.any([tokens[token_idx+1].startswith(punc) for punc in string.punctuation]):
                        valid_end.add(token_idx)
                self.cache_valid_candidates[block_idx] = {"valid_start": valid_start,
                                                          "valid_end": valid_end}


            all_start_indices = set()
            all_end_indices = set()
            all_start_and_end = set()
            for (block_idx, token_indices, vocabs), start_token_idx in zip(start_triples[0], start_indices[0]):
                if token_indices[0] not in self.cache_valid_candidates[block_idx]["valid_start"]:
                    continue
                all_start_indices.add(start_token_idx)
                for j in range(len(token_indices)):
                    if token_indices[j] not in self.cache_valid_candidates[block_idx]["valid_end"]:
                        continue
                    ngram = vocabs[:j+1]
                    ngram_pos = (start_token_idx, start_token_idx+j)
                    assert len(ngram)==ngram_pos[1]-ngram_pos[0]+1
                    if valid_func is None or valid_func(ngram):
                        if ngram_pos in pos2ngram:
                            assert pos2ngram[ngram_pos]==ngram
                        else:
                            pos2ngram[ngram_pos] = ngram
                        all_end_indices.add(start_token_idx+j)
                        all_start_and_end.add(ngram_pos)

            for (block_idx, token_indices, vocabs), end_token_idx in zip(end_triples[0], end_indices[0]):
                if token_indices[-1] not in self.cache_valid_candidates[block_idx]["valid_end"]:
                    continue
                all_end_indices.add(end_token_idx)
                for j in range(len(token_indices)):
                    if token_indices[-j-1] not in self.cache_valid_candidates[block_idx]["valid_start"]:
                        continue
                    ngram = vocabs[-j-1:]
                    ngram_pos = (end_token_idx-j, end_token_idx)
                    assert len(ngram)==ngram_pos[1]-ngram_pos[0]+1
                    if valid_func is None or valid_func(ngram):
                        if ngram_pos in pos2ngram:
                            assert pos2ngram[ngram_pos]==ngram
                        else:
                            pos2ngram[ngram_pos] = ngram
                        all_start_indices.add(end_token_idx-j)
                        all_start_and_end.add(ngram_pos)

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

        def get_predictions(mode, bm25_passages=None, prefix=""):
            predictions = {}
            start_indices, end_indices = get_initial_indices(mode, bm25_passages)

            if start_indices.shape[1]==end_indices.shape[1]==0:
                for alpha in alphas:
                    postfix = "" if alpha==0.0 else "_a={}".format(alpha)
                    predictions[prefix + "top1_ngram{}".format(postfix)] = None
                    predictions[prefix + "top1_aggregated_ngram{}".format(postfix)] = None
                return predictions

            if mode=="bm25":
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

            # if we want to normalize
            all_start_scores = softmax(all_start_scores / self.temperature, -1)
            all_end_scores = softmax(all_end_scores / self.temperature, -1)

            idx2start_score = {start_token_idx: score
                                for start_token_idx, score
                                in zip(all_start_indices, all_start_scores)}
            idx2end_score = {end_token_idx: score
                                for end_token_idx, score
                                in zip(all_end_indices, all_end_scores)}

            pos2score = {}
            ngram2score = defaultdict(list)

            # now, assign scores to possible ngrams
            for (start, end) in all_start_and_end:
                assert start in idx2start_score
                assert end in idx2end_score
                score = idx2start_score[start] + idx2end_score[end]

                pos2score[(start, end)] = score
                ngram2score[tuple(pos2ngram[(start, end)])].append(score)

            if len(pos2score)==len(ngram2score)==0:
                for alpha in alphas:
                    postfix = "" if alpha==0.0 else "_a={}".format(alpha)
                    predictions[prefix + postfix] = None
                return predictions

            assert len(pos2score)>0 and len(ngram2score)>0
            for alpha in alphas:
                def key_func(x, alpha=alpha):
                    return -np.sum(x[1]) * np.power(len(x[0]), alpha)

                top1_ngram_score_pair = min(ngram2score.items(), key=key_func)
                top1_aggregated_ngram = list(top1_ngram_score_pair[0])

                postfix = "" if alpha==0.0 else "_a={}".format(alpha)
                predictions[prefix + postfix] = top1_aggregated_ngram

            return predictions

        if self.dataset.startswith("translation"):
            assert bm25_passages is not None and oracle_passages is not None
            predictions.update(get_predictions(mode="bm25",
                                               bm25_passages=bm25_passages[:3],
                                               prefix="bm25_k=3_"))
            predictions.update(get_predictions(mode="bm25",
                                               bm25_passages=oracle_passages,
                                               prefix="oracle_"))
        else:
            try:
                index = self.dstore.index
                index_loaded = True
            except Exception:
                index_loaded = False

            if index_loaded:
                predictions.update(get_predictions(mode="default", prefix=""))

            if bm25_passages is not None:
                predictions.update(get_predictions(mode="bm25",
                                                   bm25_passages=bm25_passages[:3],
                                                   prefix="bm25_k=3_"))
                if index_loaded:
                    predictions.update(get_predictions(mode="both",
                                                    bm25_passages=bm25_passages[:3],
                                                    prefix="both_k=3_"))
        return predictions

    def get_query(self, input_text, tokenized_input):
        if tokenized_input is None:
            inputs = self.encoder(input_text)
            input_ids = inputs["input_ids"]
        else:
            input_ids = tokenized_input
        assert self.encoder.mask_token_id in input_ids
        idx = input_ids.index(self.encoder.mask_token_id)
        with torch.no_grad():
            input_tensor = torch.LongTensor([input_ids]).cuda()
        if self.model.two_mask:
            _, query = self.model.forward(input_tensor, idx, two_mask=True)
        else:
            _, query = self.model.forward(input_tensor, idx)
        return query

    def predict_multi_token(self):
        all_predictions = []
        mask = self.get_stopword_mask()
        do_restricted = self.dstore is not None and self.dstore.restricted is not None
        def valid_func(tokens):
            return np.sum(mask[tokens])==0

        for ex in tqdm(self.examples):
            if self.dataset in ["nq_full", "triviaqa_full"]:
                tokenized_input = ex["tokenized_input"]
                tokenized_answers = ex["tokenized_answers"]

                offset = 0
                while tokenized_answers[0][offset] in [5, 10, 41]:
                    offset += 1
                    if offset==len(tokenized_answers[0]):
                        offset = -1
                        break

                if offset>0:
                    prefix = self.decode(tokenized_answers[0][:offset])
                    mask_idx = tokenized_input.index(self.encoder.mask_token_id)
                    tokenized_input = tokenized_input[:mask_idx] + tokenized_answers[0][:offset] + tokenized_input[mask_idx:]
                else:
                    prefix = ""
            else:
                prefix = ""
                if self.encoder.__class__.__name__=="BertTokenizer":
                    tokenized_input = self.encoder(ex["input"].replace("<mask>", "[MASK]"))["input_ids"]
                    tokenized_answers = [self.encoder(" "+o)["input_ids"][1:-1] for o in ex["answers"]]
                else:
                    tokenized_answers = ex["tokenized_answers"]
                    tokenized_input = ex["tokenized_input"]

            if self.dataset.startswith("translation") or tokenized_answers is None:
                gts = [normalize_answer(answer) for answer in ex["answers"]]
            else:
                gts = [normalize_answer(self.decode(a)) for a in tokenized_answers]

            init_query = self.get_query(
                ex["input"] + prefix,
                tokenized_input=tokenized_input)

            dic = self.retrieve_ngram(
                init_query,
                ngram_max=20 if self.dataset.startswith("translation") else 10,
                valid_func=valid_func,
                bm25_passages=ex["bm25_passages"] if do_restricted else None,
                oracle_passages=ex["oracle_passages"] if "oracle_passages" in ex and do_restricted else None,
            )

            for key, val in dic.items():
                if key.startswith("_"):
                    continue
                ks = key.split("_")
                key = key.replace("_top1", "").replace("_aggregated", "_a")
                key = key.replace("k=", "").replace("_ngram", "").replace("_a=", "")
                if val is None:
                    acc = 0
                else:
                    acc = normalize_answer(prefix+self.decode(val)) in gts

            all_predictions.append(dic)




