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

class NPMSingle(object):
    def __init__(self, model, dstore=None, k=None, temperature=1.0):
        self.model = model
        self.k = k
        self.temperature = temperature
        self.dstore = dstore
        self.n_vocabs = len(self.model.tokenizer)

    def decode(self, ids):
        return self.model.tokenizer.decode(ids)

    def get_scores(self, queries, x):
        if type(queries)==np.ndarray:
            all_scores = np.inner(queries, x).squeeze(1) / np.sqrt(self.dstore.dimension)
        else:
            all_scores = torch.inner(queries, x).squeeze(1) / np.sqrt(self.dstore.dimension)
        return all_scores

    def get_all_scores(self, queries):
        queries = queries.detach().cpu().numpy()
        all_scores, all_indices = self.dstore.search(queries, k=self.k)
        knn_ids = self.dstore.get_block_idx_and_token(all_indices.tolist(), token_only=True)
        x = self.dstore.get_embs(all_indices)
        all_scores = self.get_scores(queries, x) / self.temperature
        return all_scores, all_indices, knn_ids

    def get_knn_scores(self,
                       queries,
                       return_context=False):

        all_scores, all_indices, knn_ids = self.get_all_scores(queries)

        if return_context:
            sorted_all_indices = all_indices[0, np.argsort(-all_scores[0])]
            assert len(sorted_all_indices)==len(knn_ids[0])

        k = all_scores.shape[1]
        assert len(knn_ids)==len(all_scores)==1 and len(knn_ids[0])==len(all_scores[0])

        probs = softmax(all_scores, -1)
        assert len(knn_ids)==1 and probs.shape[0]==1 and len(knn_ids[0])==len(probs[0])
        full_knn_scores = {}
        for vocab, p in zip(knn_ids[0], probs[0]):
            if vocab not in full_knn_scores:
                full_knn_scores[vocab] = 0
            full_knn_scores[vocab] += p

        prob = np.zeros((self.n_vocabs, ))
        for vocab, p in full_knn_scores.items():
            prob[vocab] = p

        if return_context:
            def decode_func(input_ids, token_i):
                assert token_i < len(input_ids)
                if token_i==len(input_ids)-1:
                    return self.model.tokenizer.decode(input_ids) + " " + colored("EOS", "red"), "EOS"
                retrieved_token = self.model.tokenizer.decode([input_ids[token_i+1]])
                return self.model.tokenizer.decode(input_ids[:token_i+1]) + \
                    colored(retrieved_token, "red") + \
                    self.model.tokenizer.decode(input_ids[token_i+2:]), retrieved_token

            context = self.dstore.get_context(sorted_all_indices.tolist(), decode_func)
            assert len(context)==len(all_scores[0])
            sorted_context_and_scores = sorted(zip(context, all_scores[0]),
                                               key=lambda x: -x[1])

            return prob, sorted_context_and_scores

        return prob

    def predict(self,
                input_text,
                label2id,
                return_context=False,
                max_length=256):

        assert type(input_text)==str
        if "<mask>" not in input_text:
            input_text = input_text + "{}.".format(self.model.tokenizer.mask_token)
        inputs = self.model.tokenizer.encode_plus(input_text)
        input_ids = inputs["input_ids"]

        mask_id = self.model.tokenizer.mask_token_id
        idx = input_ids.index(mask_id)

        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            assert mask_id in input_ids
            idx = input_ids.index(mask_id)

        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to("cuda")

        with torch.no_grad():
            logits, knn_queries = self.model.forward(input_ids, idx)

        if self.dstore is None:
            logits = logits.detach().cpu()
            prob = torch.softmax(logits, dim=-1).numpy()
            assert not return_context
        else:
            prob = self.get_knn_scores(knn_queries, return_context=return_context)
            if return_context:
                prob, retrieved_context = prob

        prob = np.array([np.sum(prob[label2id[label]]) for label in range(len(label2id))])
        if return_context:
            return prob, retrieved_context
        return prob

    def evaluate(self, task):
        all_predictions = []
        accs = []

        examples = task.examples
        labels = examples[0]["label_list"]
        if self.dstore is not None and task.label2syn is not None:
            label2id = self.init_label2word_id(task.label2syn)
            assert np.all([v.shape[-1]==1 for v in label2id.values()])
        else:
            labels_id = self.model.tokenizer(labels)["input_ids"]
            label2word = {i: [v] for i, v in enumerate(examples[0]["label_list"])}
            label2id = self.init_label2word_id(label2word)

        for ex in tqdm(examples):
            prob = self.predict(ex["input"], label2id)
            predicted_label = np.argmax(prob)
            accs.append(ex["label"]==predicted_label)
            all_predictions.append({"prediction": labels[predicted_label], "prob": prob.tolist()})

        print ("%s\tAccuracy=%.1f%%" % (task, 100*np.mean(accs)))
        return all_predictions

    def init_label2word_id(self, label2synonym):
        label2synonym_id = {}
        for k, v in label2synonym.items():
            synonym_id = []
            for word in v:
                tokens = self.model.tokenizer(word)["input_ids"]
                assert len(tokens)==3
                assert (tokens[0]==0 and tokens[-1]==2) or (tokens[0]==101 and tokens[-1]==102)
                tokens = tokens[1:-1]
                assert len(tokens)==1
                synonym_id.append(tokens)
            label2synonym_id[k] = np.array(synonym_id)
        return label2synonym_id

    def get_stopword_mask(self, name="stopwords", stopwords=set()):
        mask = np.zeros((self.n_vocabs, ))
        stopwords = set()
        with open("config/roberta_" + name + ".txt") as f:
            for line in f:
                stopwords.add(int(line.strip()))
        mask[np.array(list(stopwords))] = -1e10
        return mask

