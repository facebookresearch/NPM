# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import subprocess
import numpy as np
from pyserini.search.lucene import LuceneSearcher

class BM25Searcher(object):

    def __init__(self, data_dir, index_dir):
        if not os.path.exists(index_dir):
            self.build_index(data_dir, index_dir)
        self.searcher = LuceneSearcher(index_dir)
        print ("Loaded BM25 index from %s" % index_dir)

    def build_index(self, data_dir, index_dir):
        print ("Start building index for %s at %s" % (data_dir, index_dir))
        command = """python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input %s \
        --index %s \
        --generator DefaultLuceneDocumentGenerator \
            --threads 1""" % (data_dir, index_dir)
        ret_code = subprocess.run([command],
                                  shell=True,
                                  #stdout=subprocess.DEVNULL,
                                  #stderr=subprocess.STDOUT
                                  )
        if ret_code.returncode != 0:
            print("Failed to build the index")
            exit()
        else:
            print("Successfully built the index")

    def search(self, questions, k=3, is_question=False):
        is_batch = type(questions)==list
        if not is_batch:
            questions = [questions]

        results = []
        for question in questions:
            if is_question:
                question = question[:question.index("?")+1].strip()
            else:
                question = question.replace("<mask>", "_____")
            ids = [int(hit.docid) for hit in self.searcher.search(question, k=k)]
            assert len(set(ids))==len(ids), (len(results), ids)
            results.append(ids)

        if not is_batch:
            return results[0]
        return results

    def batch_search(self, input_):
        from task.task import Task
        '''
        input_ can be either
        - an instance of the class `Task`
        - a list of integers: a list of block indices you will be restricted to
        - a list of strings:  a list of inputs, if these are all you will use, so that a list of
                              block indices can be computed offline
        - a dictionary: string->a list of intergers, precomputed BM25 block indices
        - True: meaning you will use restricted search but on the fly. this will load all the embeddings
        - False or None: you will not use restricted search
        '''

        def _flatten(ids):
            if type(ids)!=list:
                return ids
            return [_id for _ids in ids for _id in _ids]

        if type(input_)==list and isinstance(input_[0], Task):
            restricted_dict = {}
            restricted = set()
            for task in input_:
                restricted_inputs = [ex["input"] for ex in task.examples]
                retrieved_ids = self.search(restricted_inputs, is_question=task.is_question)
                restricted_dict.update({
                    _input: _ids for _input, _ids in zip(restricted_inputs, retrieved_ids)
                })
                restricted |= set(_flatten(retrieved_ids))
        elif isinstance(input_, Task):
            task = input_
            restricted_inputs = [ex["input"] for ex in task.examples]
            retrieved_ids = self.search(restricted_inputs, is_question=task.is_question)
            restricted_dict = {_input: _ids for _input, _ids in zip(restricted_inputs, retrieved_ids)}
            restricted = _flatten(retrieved_ids)
            restricted = set(restricted)
        elif type(input_)==list and np.all([type(r)==str for r in input_]):
            retrieved_ids = self.search(input_)
            restricted_dict = {_input: _ids for _input, _ids in zip(input_, retrieved_ids)}
            restricted = _flatten(retrieved_ids)
            restricted = set(restricted)
        elif type(input_)==list and np.all([type(r)==int for r in input_]):
            restricted = set(self.restricted)
            restricted_dict = {}
        elif type(input_)==dict:
            restricted_dict = {k: v.copy() for k, v in input_.items()}
            restricted = set(_flatten(self.restricted_dict.values()))
        else:
            restricted = True
            restricted_dict = {}

        return restricted, restricted_dict


