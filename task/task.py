# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
from collections import defaultdict

#from task.data_loaders import *
from task.load_data import load_data, load_fuzzy_verbalizer

class Task(object):

    def __init__(self, dataname, data_dir, n_samples=0):
        examples = load_data(dataname, data_dir)

        if dataname in ["sst2", "mr", "rt", "cr", "amazon"]:
            self.label2syn = load_fuzzy_verbalizer(os.path.join(data_dir, "fuzzy_verbalizers/sst2.txt"))
        elif dataname=="agn":
            self.label2syn = load_fuzzy_verbalizer(os.path.join(data_dir, "fuzzy_verbalizers/agn.txt"))
        elif dataname=="yahoo":
            self.label2syn = load_fuzzy_verbalizer(os.path.join(data_dir, "fuzzy_verbalizers/yahoo.json"))
        else:
            self.label2syn = None

        if dataname.startswith("lama-"):
            for i, ex in enumerate(examples):
                examples[i]["ngram"] = min(4, np.min([len(a) for a in ex["tokenized_answers"]]))

        if n_samples:
            np.random.seed(0)
            examples_sample = []

            if dataname=="entity_translation":
                examples_dict = defaultdict(list)
                for ex in examples:
                    examples_dict[ex["lang"]].append(ex)
                for lang, curr_examples in examples_dict.items():
                    indices = np.random.permutation(range(len(curr_examples)))[:n_samples // 3]
                    examples_sample += [curr_examples[i] for i in indices]
                np.random.shuffle(examples_sample)
            elif dataname.startswith("lama-"):
                examples_dict = defaultdict(list)
                for ex in examples:
                    examples_dict[ex["ngram"]].append(ex)
                for n, curr_examples in examples_dict.items():
                    indices = np.random.permutation(range(len(curr_examples)))[:n_samples // 3]
                    examples_sample += [curr_examples[i] for i in indices]
                np.random.shuffle(examples_sample)
            else:
                for i in np.random.permutation(range(len(examples)))[:n_samples]:
                    examples_sample.append(examples[i])

            examples = examples_sample

        self.dataname = dataname
        self.examples = examples

        if dataname.startswith("lama-"):
            self.ngrams = [ex["ngram"] for ex in examples]
        elif dataname=="entity_translation":
            self.ngrams = [ex["lang"] for ex in examples]
        else:
            self.ngrams = None

        if dataname.startswith("lama-"):
            self.is_question = False
        elif dataname in ["kamel", "nq", "triviaqa"]:
            self.is_question = True
        else:
            self.is_question = False

    def __str__(self):
        return "Task: " + self.dataname

    def __len__(self):
        return len(self.examples)

if __name__=='__main__':
    data_dir = "data"
    #for dataname in ["agn", "yahoo", "subj", "rte", "sst2", "mr", "rt", "cr", "amazon"]:
    #    task = Task(dataname, data_dir)

    for dataname in ["lama-trex",
                     "lama-google_re",
                     #"kamel", "triviaqa", "nq"
                     ]:
        task = Task(dataname, data_dir)
        assert "input" in task.examples[0] and "answers" in task.examples[0], (dataname, task.examples[0])


