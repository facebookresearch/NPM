# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import argparse
import time
import numpy as np

import torch
from task.task import Task
from npm.npm_single import NPMSingle
from npm.npm import NPM
from npm.dstore import DataStore
from npm.model import Model, SingleModel

class NPMDemo(object):

    def __init__(self, save_dir, checkpoint_path, k, temperature,
                 remove_stopwords, remove_stopwords_except_k, single, restricted,
                 embs_consider_boundary, keep_uint8):
        start_time = time.time()
        dstore = DataStore(setting="enwiki",
                           model_dir=os.path.join(save_dir, "dstore"),
                           do_load_index=False,
                           remove_stopwords=remove_stopwords,
                           remove_stopwords_except_k=remove_stopwords_except_k,
                           restricted=restricted,
                           embs_consider_boundary=embs_consider_boundary,
                           keep_uint8=keep_uint8
                           )
        model_class = SingleModel if single else Model
        model = model_class(checkpoint_path=checkpoint_path)
        print ("Finish loading the model (%dsec)" % (time.time()-start_time))

        npm_class = NPMSingle if single else NPM
        npm = npm_class(model=model, dstore=dstore, k=k, temperature=temperature)

        mask = npm.get_stopword_mask()
        def valid_func(tokens):
            return np.sum(mask[tokens])==0

        self.npm = npm
        self.valid_func = valid_func

    def predict(self, text):
        if "<mask>" not in text:
            text = text.strip() + "<mask>."
        predicted = self.npm.predict_span(text,
                                          ngram_max=10,
                                          valid_func=self.valid_func,
                                          alphas=[0.0])["a=0.0"]
        return self.npm.decode(predicted)

    def generate(self, text, num_tokens=20, num_masked_tokens=20):
        assert "<mask>" not in text
        for _ in range(num_tokens):
            input_text = text + "<mask>"*num_masked_tokens
            predicted = self.npm.predict_span(input_text,
                                              ngram_max=10,
                                              valid_func=self.valid_func,
                                              alphas=[0.0])["a=0.0"]
            predicted = self.npm.decode(predicted)
            text += predicted
        return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="save")
    parser.add_argument('--checkpoint_path', type=str, default="npm")
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--remove_stopwords_except_k", type=int, default=None)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--restricted", action="store_true")

    parser.add_argument("--embs_consider_boundary", action="store_true", default=True)
    parser.add_argument("--keep_uint8", action="store_true")

    args = parser.parse_args()
    npm = NPMDemo(save_dir=args.save_dir,
                  checkpoint_path=args.checkpoint_path,
                  k=args.k,
                  temperature=args.temperature,
                  remove_stopwords=args.remove_stopwords,
                  remove_stopwords_except_k=args.remove_stopwords_except_k,
                  single=args.single,
                  restricted=args.restricted,
                  embs_consider_boundary=args.embs_consider_boundary,
                  keep_uint8=args.keep_uint8)

    def predict(text):
        start_time = time.time()
        predicted = npm.predict(text)
        print ("(Took %.2fs)" % (time.time()-start_time))
        return predicted

    input_text = "Hagios Demetrios is located in"
    print (predict(input_text))

    print (npm.generate(input_text))


if __name__=='__main__':
    main()



