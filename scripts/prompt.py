# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import argparse
import time

import torch
from task.task import Task
from npm.npm_single import NPMSingle
from npm.npm import NPM
from npm.dstore import DataStore, DataStoreUnion
from npm.model import Model, SingleModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dataset', type=str, default="all")
    parser.add_argument('--corpus_data', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default="roberta-large")
    parser.add_argument('--save_dir', type=str, default="save")

    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n_samples', type=int, default=3000)
    parser.add_argument("--remove_stopwords", action="store_true")

    parser.add_argument("--single", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--restricted", action="store_true")

    args = parser.parse_args()
    print (args)

    if args.restricted:
        assert "+" not in args.eval_dataset
        task = Task(args.eval_dataset, "data", n_samples=args.n_samples)

    start_time = time.time()

    if args.corpus_data is None:
        dstore = None
    else:
        dstore_class = DataStoreUnion if "+" in args.corpus_data else DataStore
        dstore = dstore_class(setting=args.corpus_data,
                              model_dir=os.path.join(args.save_dir, "dstore"),
                              do_load_index=not args.restricted,
                              remove_stopwords=args.remove_stopwords,
                              restricted=task if args.restricted else None,
                              )
        print ("Finish loading the datastore (%dsec)" % (time.time()-start_time))

    def add_postfix(corpus_data, postfix):
        return corpus_data.replace("+", postfix + "+") + postfix

    if args.remove_stopwords:
        args.corpus_data = add_postfix(args.corpus_data, ":no_stopwords")

    model_class = SingleModel if args.single else Model
    model = model_class(checkpoint_path=args.checkpoint_path)
    print ("Finish loading the model")

    if args.eval_dataset is None:
        return

    npm_class = NPMSingle if args.single else NPM
    npm = npm_class(model=model, dstore=dstore, k=args.k, temperature=args.temperature)
    for eval_dataset in args.eval_dataset.split("+"):
        # loading the task data
        if not args.restricted:
            task = Task(eval_dataset, "data", n_samples=args.n_samples)

        save_dir = os.path.join(args.save_dir, "results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if args.open:
            if args.single:
                raise NotImplementedError("NPM Single does not support open-set tasks.")
            all_predictions = npm.evaluate_open(task)
        else:
            all_predictions = npm.evaluate(task)

        save_path = os.path.join(save_dir, "{}{}{}{}{}.txt".format(
            eval_dataset,
            "_c={}".format(args.corpus_data) if dstore is not None else "",
            "_k={}".format(args.k) if dstore is not None else "",
            "_t={}".format(args.temperature) if dstore is not None else "",
            "_restricted" if args.restricted else ""
        ))

        with open(save_path, "w") as f:
            for pred in all_predictions:
                f.write(json.dumps(pred)+"\n")

if __name__=='__main__':
    main()



