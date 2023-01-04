# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import time
import string
import numpy as np

from scipy.linalg import block_diag
from collections import Counter, defaultdict

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
mask_id = tokenizer.mask_token_id

idx2word = {idx: tokenizer._decode(idx) for idx in range(len(tokenizer.encoder))}
punctuation = set()
subwords = set()
for idx, word in idx2word.items():
    if word in ["!", ",", ".", ":", "?"]:
        punctuation.add(idx)
    elif word[0]!=" " and word not in string.punctuation:
        subwords.add(idx)

def find_mapping(input_ids, attention_mask):
    blocks = []
    for _input_ids, _attention_mask in zip(input_ids, attention_mask):
        blocks.append(_input_ids[:np.sum(_attention_mask)])

    datastore = defaultdict(list)
    mapping = defaultdict(list)
    N = 1

    def construct_datastore(length_list=range(1, N+1)):
        for block_idx, block in enumerate(blocks):
            for token_idx, token in enumerate(block):
                for L in length_list:
                    key = block[token_idx:token_idx+L]
                    if 0 in key or 2 in key:
                        continue
                    if len(key)==L:
                        datastore[tuple(key)].append((block_idx, token_idx))

    construct_datastore()

    for block_idx, block in enumerate(blocks):
        token_idx = 0
        for token_idx, token in enumerate(block):
            if token in [0, 2]:
                continue
            # let's remove starting from subwords
            if (token_idx==0 or block[token_idx-1]!=0) and token in subwords:
                continue

            if token in punctuation and np.random.random()<0.9:
                continue

            pos = (block_idx, token_idx)

            for L in range(1, 11):
                key = block[token_idx:token_idx+L]
                if 0 in key or 2 in key or len(key)<L:
                    break
                # let's remove if it is 2+ gram and start from end-of-sentence punctuation
                if L>=2 and token in punctuation:
                    break
                # let's remove if the next token is the subword
                if token_idx+L<=len(block)-1 and block[token_idx+L] in subwords:
                    continue

                assert len(key)==L
                key = tuple(key)
                if key not in datastore:
                    # further construct the datastore
                    construct_datastore(length_list=range(L, 2*L))

                assert key in datastore
                assert pos in datastore[key]

                candidates = [t for t in datastore[key] if block_idx!=t[0]]
                if len(candidates)==0:
                    break
                else:
                    mapping[pos].append((candidates, L, key))

    return mapping


def mask_spans(data_path, masking_ratio=0.4, p=0.2):
    np.random.seed(2022)
    start_time = time.time()
    out_data_path = data_path.replace(".jsonl", "_mr{}_p{}.jsonl".format(masking_ratio, p))

    tot = 0
    line_idx = 0
    start_time = time.time()

    with open(data_path, "r") as f:
        with open(out_data_path, "w") as f_w:

            for line in f:
                dp = json.loads(line)
                mapping = find_mapping(dp["input_ids"], dp["attention_mask"])

                # this is to sample spans to mask out
                ngram2spans = defaultdict(list)
                pos2dependent_keys = defaultdict(list)

                for (i, j), triples in mapping.items():
                    for (candidates, n, key) in triples:
                        ngram2spans[min(10, n)].append((i, j, n))
                        for (other_i, other_j) in candidates:
                            for k in range(n):
                                pos2dependent_keys[(other_i, other_j+k)].append((i, j, n))

                input_ids = np.array(dp["input_ids"])
                attention_mask = np.array(dp["attention_mask"])
                BS, length = input_ids.shape
                ngram2spans = {k: np.random.permutation(v).tolist() for k, v in ngram2spans.items()}

                masked_input_ids = input_ids.copy()
                mask_budget = int(np.sum(attention_mask)*masking_ratio)
                mask_cnts = []

                finish_masking = False
                masked_triples = set()
                masked_ngram_counter = Counter()
                n_masks_counter = Counter()
                mask_list = defaultdict(list)

                for n in np.random.geometric(p=p, size=(mask_budget, )):
                    if n>10:
                        continue

                    while True:
                        if False:
                            if len(all_spans)==0:
                                finish_masking = True
                                break
                            i, j, ngram = all_spans[-1]
                            all_spans = all_spans[:-1]
                        else:
                            while n>0 and (n not in ngram2spans or len(ngram2spans[n])==0):
                                assert n>0
                                n -= 1
                            if n==0:
                                finish_masking = True
                                break
                            i, j, ngram = ngram2spans[n][-1]
                            ngram2spans[n] = ngram2spans[n][:-1]

                        # don't mask from the same sequence too many times
                        if n_masks_counter[i] + 1 > 64:
                            continue

                        # tokens-to-be-masked shouldn't be already
                        # masked out
                        if np.sum(masked_input_ids[i, j:j+ngram]==mask_id)>0:
                            # print ("skipping because some of them are already masked out")
                            continue

                        # if the same ngram has been masked out too much, then skip
                        freq = masked_ngram_counter[tuple(masked_input_ids[i,j:j+ngram])]
                        if freq >= 10:
                            # print ("skipping", tokenizer.decode(masked_input_ids[i,j:j+ngram]))
                            continue

                        '''
                        # see if ids covering this position is fine
                        candidates, n1, _ = [triple for triple in mapping[(i, j)] if triple[1]==ngram][0]
                        assert n1==ngram
                        covered = False
                        for (other_i, other_j) in candidates:
                            if np.sum(masked_input_ids[other_i, other_j:other_j+ngram]==mask_id)==0:
                                covered = True
                                break
                        if not covered:
                            continue

                        # see if ids covered by this position is fine
                        not_covered_found = False
                        dependencies = set()
                        for k in range(ngram):
                            dependencies |= set(pos2dependent_keys[(i, j+k)])
                        dependencies &= masked_triples

                        for (other_i, other_j, other_n) in dependencies:
                            # let's make sure there're other ngrams that cover
                            # (other_i, other_j, other_n)
                            other_candidates, n1, _ = \
                                [triple for triple in mapping[(other_i, other_j)]
                                if triple[1]==other_n][0]
                            assert n1==other_n

                            covered = 0
                            for (another_i, another_j) in other_candidates:
                                if another_i==i and \
                                        len(set(range(another_j, another_j+n1)) & set(range(j, j+ngram))) > 0:
                                    pass
                                elif np.sum(masked_input_ids[another_i, another_j:another_j+n1]==mask_id)==0:
                                    covered += 1
                                    break

                            if covered==0:
                                not_covered_found = True
                                break

                        if not_covered_found:
                            #print ("skipping 2")
                            continue
                        '''
                        break

                    if finish_masking:
                        break

                    masked_triples.add((i, j, ngram))
                    mask_list[i].append((j, ngram))
                    masked_ngram_counter[tuple(masked_input_ids[i,j:j+ngram])] += 1
                    masked_input_ids[i,j:j+ngram] = mask_id
                    n_masks_counter[i] += 1

                    mask_cnts.append(ngram)
                    # masking_ngram_list.append(ngram)
                    if np.sum(mask_cnts) >= mask_budget:
                        break

                #t2 = time.time()
                curr_ratio = np.sum(masked_input_ids==mask_id)/np.sum(attention_mask)

                if curr_ratio < 0.05:
                    # print ("skipping because not much to mask out")
                    continue

                # masking_ratio_list.append(curr_ratio)

                input_ids = input_ids.tolist()
                masked_input_ids = masked_input_ids.tolist()
                attention_mask = attention_mask.tolist()

                merged_masked_input_ids = []
                merged_attention_mask = []
                merged_labels = []

                for i, (curr_input_ids, curr_masked_input_ids, curr_attention_mask) in enumerate(zip(
                        input_ids, masked_input_ids, attention_mask)):

                    curr_merged_masked_input_ids = []
                    curr_merged_attention_mask = []
                    curr_merged_labels = []

                    curr_mask_list = sorted(mask_list[i])
                    '''
                    offset = 0
                    while mask_id in curr_masked_input_ids[offset:]:
                        start_idx = offset + curr_masked_input_ids[offset:].index(mask_id)
                        end_idx = start_idx
                        while end_idx+1<len(curr_masked_input_ids) and curr_masked_input_ids[end_idx+1]==mask_id:
                            end_idx += 1

                        assert curr_input_ids[offset:start_idx]==curr_masked_input_ids[offset:start_idx]

                        curr_merged_masked_input_ids += curr_input_ids[offset:start_idx]
                        curr_merged_masked_input_ids += [mask_id]

                        curr_merged_attention_mask += curr_attention_mask[offset:start_idx]
                        curr_merged_attention_mask += [1]

                        # we will put labels for the number of mask tokens only
                        curr_merged_labels.append(curr_input_ids[start_idx:end_idx+1])
                        assert 0 not in curr_input_ids[start_idx:end_idx+1]
                        assert 2 not in curr_input_ids[start_idx:end_idx+1]
                        offset = end_idx+1
                    '''
                    offset = 0
                    for (start_idx, ngram) in curr_mask_list:
                        end_idx = start_idx + ngram
                        assert curr_input_ids[offset:start_idx]==curr_masked_input_ids[offset:start_idx]

                        curr_merged_masked_input_ids += curr_input_ids[offset:start_idx]
                        curr_merged_masked_input_ids += [mask_id]

                        curr_merged_attention_mask += curr_attention_mask[offset:start_idx]
                        curr_merged_attention_mask += [1]

                        # we will put labels for the number of mask tokens only
                        curr_merged_labels.append(curr_input_ids[start_idx:end_idx])
                        assert 0 not in curr_input_ids[start_idx:end_idx]
                        assert 2 not in curr_input_ids[start_idx:end_idx]
                        offset = end_idx

                    curr_merged_masked_input_ids += curr_input_ids[offset:]
                    curr_merged_attention_mask += curr_attention_mask[offset:]

                    assert len(curr_merged_masked_input_ids)==len(curr_merged_attention_mask)

                    while len(curr_merged_masked_input_ids) < length:
                        curr_merged_masked_input_ids.append(0)
                        curr_merged_attention_mask.append(0)


                    assert curr_merged_attention_mask[0]

                    merged_masked_input_ids.append(curr_merged_masked_input_ids)
                    merged_attention_mask.append(curr_merged_attention_mask)
                    merged_labels.append(curr_merged_labels)

                '''
                str_input_ids = [",".join([str(i) for i in _input_ids])
                                for _input_ids in input_ids]
                str_masked_input_ids = [",".join([str(i) for i in _input_ids])
                                        for _input_ids in masked_input_ids]
                for i, labels in enumerate(merged_labels):
                    for l in labels:
                        l = ",".join([str(k) for k in l])
                        assert np.any([l in str_input_ids for str_input_ids in str_input_ids])
                        assert np.any([l in str_input_ids for str_input_ids in str_masked_input_ids])
                '''
                dp["masked_input_ids"] = masked_input_ids
                dp["merged_masked_input_ids"] = merged_masked_input_ids
                dp["merged_attention_mask"] = merged_attention_mask
                dp["merged_labels"] = merged_labels

                #t3 = time.time()
                #t_mapping += t1-t0
                #t_masking += t2-t1
                #t_merging += t3-t2

                assert dp["input_ids"]!=dp["masked_input_ids"]
                f_w.write(json.dumps(dp)+"\n")

                tot += 1
                if tot % 1000 == 0:
                    print ("Finish saving %dK lines for %s (%dmin)" % (
                        tot/1000, data_path, (time.time()-start_time)/60))

    print ("Finish %s (%dmin)" % (data_path, (time.time()-start_time)/60))


