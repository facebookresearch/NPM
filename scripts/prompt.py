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
    parser.add_argument('--checkpoint_path', type=str, default="npm")
    parser.add_argument('--save_dir', type=str, default="save")

    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n_samples', type=int, default=3000)
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--remove_stopwords_except_k", type=int, default=None)

    parser.add_argument("--single", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--restricted", action="store_true")

    # for ablations
    parser.add_argument("--load_all_embs", action="store_true", default=True)
    parser.add_argument("--embs_consider_boundary", action="store_true", default=True)
    parser.add_argument("--keep_uint8", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    print (args)

    if args.restricted and not args.load_all_embs:
        tasks = []
        for eval_dataset in args.eval_dataset.split("+"):
            task = Task(eval_dataset, "data", n_samples=args.n_samples)
            tasks.append(task)

    start_time = time.time()

    if args.corpus_data is None:
        dstore = None
    else:
        dstore_class = DataStoreUnion if "+" in args.corpus_data else DataStore
        dstore = dstore_class(setting=args.corpus_data,
                              model_dir=os.path.join(args.save_dir, "dstore"),
                              do_load_index=not args.restricted,
                              remove_stopwords=args.remove_stopwords,
                              remove_stopwords_except_k=args.remove_stopwords_except_k,
                              restricted=(True if args.load_all_embs else tasks) if args.restricted else None,
                              embs_consider_boundary=args.embs_consider_boundary,
                              keep_uint8=args.keep_uint8
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
    npm = npm_class(model=model,
                    dstore=dstore,
                    k=args.k,
                    temperature=args.temperature)

    for dataset_idx, eval_dataset in enumerate(args.eval_dataset.split("+")):
        # loading the task data
        if args.restricted and not args.load_all_embs:
            task = tasks[dataset_idx]
        else:
            task = Task(eval_dataset, "data", n_samples=args.n_samples)

        if args.debug:
            import numpy as np
            from task.utils_eval import normalize_answer

            # evaluate on a subset of examples where BM25 is successful.
            if args.load_all_embs:
                _, restricted_dict = dstore.searcher.batch_search(task)
            else:
                restricted_dict = dstore.restricted_dict
            psg_id_to_raw_text = {}
            for psgs in restricted_dict.values():
                for psg in psgs:
                    if psg not in psg_id_to_raw_text:
                        psg_id_to_raw_text[psg] = normalize_answer(npm.decode(dstore.input_ids[psg]))

            included = []
            for i, ex in enumerate(task.examples):
                psgs = restricted_dict[ex["input"]]
                psgs = [psg_id_to_raw_text[psg] for psg in psgs]
                answers = [normalize_answer(answer) for answer in ex["answers"]]
                if np.any([answer in psg for answer in answers for psg in psgs]):
                    included.append(i)
            print ("Evaluating %d->%d examples..." % (len(task.examples), len(included)))
            task.examples = [task.examples[i] for i in included]
            if task.ngrams is not None:
                task.ngrams = [task.ngrams[i] for i in included]

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



