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
import pickle as pkl

from tqdm import tqdm
from scipy.special import log_softmax

from collections import defaultdict, Counter

#import tracemalloc
import os
import psutil
# tracemalloc.start()
pid = os.getpid()
python_process = psutil.Process(pid)

def print_mem_use():
    #memUse = tracemalloc.get_traced_memory()[1]/(1024*1024*1024)
    memUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use: %.1fGB' % memUse)

def load_embs(embed_path, dstore_size, dimension):
    assert os.path.exists(embed_path), embed_path
    return np.memmap(embed_path,
                     dtype=np.float16,
                     mode="r",
                     shape=(dstore_size, dimension))

class DataStore(object):
    def __init__(self,
                 setting=None,
                 data_path=None,
                 model_dir=None,
                 do_load_data=True,
                 do_load_embeds=True,
                 do_load_index=True,
                 dimension=1024,
                 ncentroids=4096,
                 code_size=64,
                 probe=8,
                 num_keys_to_add_at_a_time=1000000,
                 remove_stopwords=False,
                 restricted=None,
                 consider_string_boundary=True,
                 cuda=True,
                 embs_consider_boundary=False,
                 keep_uint8=False
                 ):

        base_dir = "corpus"
        if setting in ["enwiki-0", "enwiki-2022-0"]:
            data_path = os.path.join(base_dir, setting[:-2], "0.npy")
            if model_dir is not None:
                model_dir = os.path.join(model_dir, setting)
        elif setting in ["enwiki", "enwiki-2022"]:
            assert remove_stopwords, remove_stopwords
            data_path=[os.path.join(base_dir, setting, "{}.npy".format(idx)) for idx in range(20)]
            if model_dir is not None:
                model_dir=[os.path.join(model_dir, "{}-{}".format(setting, idx)) for idx in range(20)]
                ncentroids *= 8
        elif setting in ["cc_news", "imdb", "amazon", "subj"]:
            data_path = os.path.join(base_dir, setting, "text.npy")
            if model_dir is not None:
                model_dir = os.path.join(model_dir, setting)
        else:
            raise NotImplementedError(setting)

        self.setting = setting
        self.dimension = dimension
        self.ncentroids = ncentroids
        self.code_size = code_size
        self.probe = probe
        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time
        self.remove_stopwords = remove_stopwords
        self.restricted = restricted
        self.consider_string_boundary = consider_string_boundary
        self.cuda = cuda
        self.embs_consider_boundary = embs_consider_boundary
        self.keep_uint8 = keep_uint8

        '''
        restricted can be either
        - an instance of the class `Task`
        - a list of integers: a list of block indices you will be restricted to
        - a list of strings:  a list of inputs, if these are all you will use, so that a list of
                              block indices can be computed offline
        - a dictionary: string->a list of intergers, precomputed BM25 block indices
        - True: meaning you will use restricted search but on the fly. this will load all the embeddings
        - False or None: you will not use restricted search
        '''

        if self.restricted:
            from npm.searcher import BM25Searcher
            data_dir = os.path.join(base_dir, setting)
            index_dir = os.path.join(base_dir, setting + "-index")
            self.searcher = BM25Searcher(data_dir, index_dir)
            self.restricted, self.restricted_dict = self.searcher.batch_search(self.restricted)

        self.load_restricted = self.restricted and type(self.restricted)!=bool
        print ("load_restricted:", self.load_restricted)

        if do_load_data:
            self.load_data(data_path)

        print_mem_use()

        if do_load_embeds:
            assert model_dir is not None
            assert do_load_data
            self.load_embeds(model_dir)

        print_mem_use()

        if do_load_index:
            assert model_dir is not None
            self.load_index(model_dir)

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

    def load_data(self, data_path):
        self.input_ids = []
        self.token_idx_to_block_idx = []
        self.token_idx_to_local_idx = []
        self.emb_token_idx_to_orig_block_idx = []
        self.orig_block_idx_to_emb_token_idx = []

        # for debugging, later we can delete this
        self.orig_block_idx_to_valid_start = {}
        self.orig_block_idx_to_valid_end = {}

        stopwords = self.load_stopwords()
        dstore_size_list = []

        if type(data_path)==list:
            data_paths = data_path
        else:
            data_paths = [data_path]

        offset = 0
        global_dstore_size = 0
        global_true_dstore_size = 0
        true_dstore_size_list = []

        if self.load_restricted:
            self.orig_emb_token_indices_valid = set()

        print_mem_use()

        for data_path_idx, _data_path in enumerate(data_paths):
            input_ids = np.load(_data_path)

            start_end_pairs = np.load(_data_path.replace(".npy", "_blocks.npy"))
            if self.consider_string_boundary:
                with open(_data_path.replace(".npy", "_valid.pkl"), "rb") as f:
                    valid_candidates = pkl.load(f)

            dstore_size = 0
            true_dstore_size = 0
            offset_block = 0 if self.input_ids is None else len(self.input_ids)

            for block_idx, (valid_start, valid_end) in enumerate(tqdm(valid_candidates)):
                start = start_end_pairs[block_idx]
                end = start_end_pairs[block_idx+1] if block_idx<len(start_end_pairs)-1 else len(input_ids)
                curr_input_ids = input_ids[start:end]
                if not self.keep_uint8:
                    curr_input_ids = curr_input_ids.tolist()
                is_valid = (not self.load_restricted) or offset in self.restricted
                valid_idxs = set(valid_start) | set(valid_end)
                curr_dstore_size = 0

                for i, curr_token in enumerate(curr_input_ids):
                    if self.remove_stopwords and curr_token in stopwords:
                        continue
                    if self.embs_consider_boundary and i not in valid_idxs:
                        continue
                    elif curr_token in [0, 2]:
                        continue
                    if is_valid:
                        self.token_idx_to_block_idx.append(len(self.input_ids))
                        self.token_idx_to_local_idx.append(i)
                    curr_dstore_size += 1

                self.orig_block_idx_to_emb_token_idx.append(global_true_dstore_size)

                if is_valid and self.load_restricted:
                    for j in range(curr_dstore_size):
                        self.emb_token_idx_to_orig_block_idx.append(offset)
                        self.orig_emb_token_indices_valid.add(global_dstore_size+j)

                if is_valid and self.consider_string_boundary:
                    self.orig_block_idx_to_valid_start[offset] = valid_start
                    self.orig_block_idx_to_valid_end[offset] = valid_end

                dstore_size += curr_dstore_size
                global_dstore_size += curr_dstore_size
                if is_valid:
                    true_dstore_size += curr_dstore_size
                    global_true_dstore_size += curr_dstore_size
                self.input_ids.append(curr_input_ids)
                offset += 1

            dstore_size_list.append(dstore_size)
            true_dstore_size_list.append(true_dstore_size)
            print ("Finished reading %.3fM tokens from %s" % (dstore_size/1000000, _data_path))
            print_mem_use()

        self.orig_block_idx_to_emb_token_idx.append(global_true_dstore_size)

        self.token_idx_to_block_idx = np.array(self.token_idx_to_block_idx)
        self.token_idx_to_local_idx = np.array(self.token_idx_to_local_idx, dtype=np.uint8)
        self.dstore_size_list = dstore_size_list
        self.dstore_size = np.sum(dstore_size_list)
        self.true_dstore_size_list = true_dstore_size_list
        self.true_dstore_size = np.sum(true_dstore_size_list)

    def load_embeds(self, model_dir):
        postfix = "_wo_stopwords" if self.remove_stopwords else ""
        if type(model_dir)==list:
            self.embs = []
            for _model_dir, dstore_size in zip(model_dir, self.dstore_size_list):
                embed_path = os.path.join(_model_dir,
                                            "embeddings{}.float16.npy".format(postfix))
                print ("Start loading the embed from %s with (%d, %d)..." % (embed_path.split("/")[-2], dstore_size, self.dimension))
                curr_emb = load_embs(embed_path, dstore_size, self.dimension)
                self.embs.append(curr_emb)
        else:
            embed_path = os.path.join(model_dir, "embeddings{}.float16.npy".format(postfix))
            print ("Start loading the embed with (%d, %d)..." % (self.dstore_size, self.dimension))
            self.embs = load_embs(embed_path, self.dstore_size, self.dimension)

        if self.load_restricted:
            if type(self.embs)==list:
                offset = 0
                for i, (emb, dstore_size) in enumerate(zip(self.embs, self.dstore_size_list)):
                    assert i>0 or offset==0
                    curr_restricted = sorted([j-offset for j in self.orig_emb_token_indices_valid
                                              if offset<=j<offset+dstore_size])
                    if len(curr_restricted)>0:
                        curr_restricted = np.array(curr_restricted)
                        self.embs[i] = emb[curr_restricted]
                        assert self.embs[i].shape[0]==self.true_dstore_size_list[i]
                    else:
                        assert self.true_dstore_size_list[i]==0
                    offset += dstore_size

                assert np.sum([emb.shape[0] for emb in self.embs])==self.true_dstore_size
            else:
                self.embs = self.embs[sorted(self.orig_emb_token_indices_valid)]
                assert self.embs.shape[0]==len(self.orig_emb_token_indices_valid)==self.true_dstore_size
                print ("Finished loading embs.shape=%s" % str(self.embs.shape))

    def load_index(self, model_dir):
        if type(model_dir)==list:
            model_dir = model_dir[-1] + ".combined"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        index_path = os.path.join(model_dir.replace("*", ""),
                                  "embeddings{}.faiss_index_IP".format("_wo_stopwords" if self.remove_stopwords else ""))
        print ("Starting loading %s" % index_path)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.index.nprobe = self.probe
        else:
            print ("No index found from %s -- start building index!" % index_path)
            if not os.path.exists(index_path + ".trained"):
                self._train_index(index_path)
            self.index = self._add_keys(index_path)

    def get_embs(self, indices):
        if type(self.embs)==list:
            if type(indices) in [int, np.int64]:
                offset = 0
                for idx, dstore_size in enumerate(self.true_dstore_size_list):
                    if offset <= indices < offset+dstore_size:
                        return self.embs[idx][indices-offset].astype(np.float32)
                    offset += dstore_size
                # it should be returned
                raise NotImplementedError()
            else:
                embs = []
                for _indices in indices:
                    embs.append(self.get_embs(_indices))
                return np.stack(embs, 0)

        return self.embs[indices].astype(np.float32)

    def get_block_idx_and_token(self, i, token_only=False):
        if type(i)==list:
            return [self.get_block_idx_and_token(j, token_only=token_only) for j in i]
        if token_only:
            return self._get_token(i)[1]
        return self._get_token(i)

    def get_context(self, i, decode_func):
        if type(i)==list:
            return [self.get_context(j, decode_func) for j in i]

        block_i, token_i = self.token_idx_to_block_idx[i], self.token_idx_to_local_idx[i]
        input_ids = self.input_ids[block_i]
        #input_ids = self.blocks[block_i]["input_ids"]
        return decode_func(input_ids, token_i)

    def get_frequency(self, tokens):
        return [self.token_counter.get(token, 0) for token in tokens]

    def search(self, query_embs, k=4096):
        all_scores, all_indices = self.index.search(query_embs.astype(np.float32), k)
        return all_scores, all_indices

    def get_knn_scores(self, query_emb, indices, distance_type="l2", temperature=1.0):
        '''
        query_emb: [batch_size, dimension]
        indices: [batch_size, k]
        self.get_embs(indices): [batch_size, k, dimension]
        distances: [batch_size, k]
        '''
        scores = np.squeeze(np.matmul(
                self.get_embs(indices), np.expand_dims(query_emb, -1)), -1)
        scores /= np.sqrt(self.dimension)
        knn_scores = np.exp(scores / temperature)
        return knn_scores

    def get_prediction_and_knn_prob(self,
                                    knn_scores,
                                    indices,
                                    gt_token,
                                    gt_token_idx=None,
                                    exclude_level=None):
        '''
        gt_token_idx and exclude_level are only specified
        if the evaluation data and the corpus data are the same
        '''
        if gt_token_idx is not None and exclude_level is not None:
            assert exclude_level in ["token", "block", "doc"]
            gt_block, _gt_token = self.get_block_idx_and_token(gt_token_idx)
            assert gt_token==_gt_token
        else:
            assert gt_token_idx is None and exclude_level is None

        score_per_token = self.init_score_per_token.copy()
        included_tokens = set()
        tot = self.init_score_sum
        for idx, knn_score in zip(indices, knn_scores):
            block, pred = self._get_token(idx)
            if pred=="EOS":
                continue
            if exclude_level is not None:
                if exclude_level=="token" and idx==gt_token_idx:
                    continue
                elif exclude_level=="block" and gt_block==block:
                    continue
                elif exclude_level=="doc" and self.block_idx_to_doc_idx[block]==self.block_idx_to_doc_idx[gt_block]:
                    continue
            included_tokens.add(pred)
            score_per_token[pred] += knn_score
            tot += knn_score

        sorted_tokens = sorted(included_tokens, key=lambda x: -score_per_token[x])
        knn_prob = score_per_token[gt_token] / tot
        return sorted_tokens, np.log(knn_prob)

    def _get_token(self, token_idx):
        block_i, token_i = self.token_idx_to_block_idx[token_idx], self.token_idx_to_local_idx
        #input_ids = self.blocks[block_i]["input_ids"]
        input_ids = self.input_ids[block_i]
        assert token_i < len(input_ids)
        token_i = input_ids[token_i]
        return block_i, token_i

    def _get_token_position(self, token_idx, ngram_before=1, ngram_after=1):
        if type(token_idx)==list:
            return [self._get_token_position(_token_idx,
                                             ngram_before=ngram_before,
                                             ngram_after=ngram_after)
                    for _token_idx in token_idx]
        block_i, token_i = self.token_idx_to_block_idx[token_idx], self.token_idx_to_local_idx[token_idx]
        #input_ids = self.blocks[block_i]["input_ids"]
        input_ids = self.input_ids[block_i]
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

    def _train_index(self, index_path):
        start = time.time()
        quantizer = faiss.IndexFlatIP(self.dimension)
        start_index = faiss.IndexIVFPQ(quantizer,
                                        self.dimension,
                                        self.ncentroids,
                                        self.code_size,
                                        8)
        start_index.nprobe = self.probe
        np.random.seed(1)

        print ("Sampling for training the index (from %d tokens)" % (self.true_dstore_size))
        if type(self.embs)==list:
            sampled_embs = []
            for emb in tqdm(self.embs):
                sampled_indices = np.random.choice(np.arange(emb.shape[0]),
                                                   size=[min(1000000, emb.shape[0])],
                                                   replace=False)
                sampled_embs.append(emb[sampled_indices])
            print ("Finish sampling; now concatenating...")
            sampled_embs = np.concatenate(sampled_embs, 0).astype(np.float32)
        else:
            random_sample = np.random.choice(np.arange(self.true_dstore_size),
                                             size=[min(1000000, self.true_dstore_size)],
                                             replace=False)
            sampled_embs = self.get_embs(random_sample) # already converted into float32

        print ("Training examples sampled; now start training...")

        if self.cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False

            # Train on GPU and back to CPU
            gpu_index.train(sampled_embs)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(sampled_embs)

        print ('Training took {} s'.format(time.time() - start))
        faiss.write_index(start_index, index_path + ".trained")

    def _add_keys(self, index_path):
        index = faiss.read_index(index_path + ".trained")
        start_time = time.time()

        if type(self.embs)==list:
            tot = 0
            for i, emb in enumerate(self.embs):
                dstore_size = emb.shape[0]
                start = 0
                while start < dstore_size:
                    end = min(dstore_size, start + self.num_keys_to_add_at_a_time)
                    to_add = self.get_embs(range(start, end), i)
                    index.add(to_add)
                    tot += (end-start)
                    start = end
                print ('idx=%d finished -- Added %d tokens (%d min)' % (
                    i, tot, (time.time()-start_time)/60))
                faiss.write_index(index, index_path)
                print ("Finish writing index (%dmin)" % ((time.time()-start_time)/60))
        else:
            start = 0
            while start < self.true_dstore_size:
                end = min(self.true_dstore_size, start + self.num_keys_to_add_at_a_time)
                to_add = self.get_embs(range(start, end)).copy()
                index.add(to_add)
                start = end

                if (start % 1000000) == 0:
                    print ('Added %d tokens (%d min)' % (start, (time.time()-start_time)/60))
                    faiss.write_index(index, index_path)

        print ("Adding total %d keys" % start)
        print ('Adding took {} s'.format(time.time() - start_time))
        faiss.write_index(index, index_path)
        return index

class DataStoreUnion(DataStore):
    def __init__(self, setting, **kwargs):
        self.dstores = []
        for _setting in setting.split("+"):
            self.dstores.append(DataStore(setting=_setting, **kwargs))
        self.dimension = self.dstores[0].dimension

    def search(self, queries, k):
        all_scores, all_indices = [], []
        for dstore_idx, dstore in enumerate(self.dstores):
            scores, indices = dstore.search(queries, k)
            all_scores.append(scores)
            all_indices.append(indices)

        return np.stack(all_scores, 0), np.stack(all_indices, 0)

    def get_block_idx_and_token(self, indices, token_only):
        assert token_only # not implemented otherwise
        assert len(indices)==len(self.dstores)
        tokens = None
        for dstore, _indices in zip(self.dstores, indices):
            curr_tokens = dstore.get_block_idx_and_token(_indices, token_only=True)
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


