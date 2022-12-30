# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import argparse
import numpy as np

from collections import defaultdict
from prettytable import PrettyTable
from task.task import Task
from task.utils_eval import normalize_answer

def load_output_file(output_file):
    predictions = []
    with open(output_file, "r") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions

def main(args):

    if args.closed:

        datasets = ["agn", "yahoo", "subj", "sst2", "mr", "rt", "cr", "amazon", "rte"]
        tasks = []
        for dataset in datasets:
            tasks.append(Task(dataset, "data", n_samples=3000))

        pt = PrettyTable()
        pt.field_names = ["Model"] + datasets
        pt.align["Model"] = "l"
        for dataset in datasets:
            pt.align[dataset] = "r"

        row = ["RoBERTa"]
        for dataset, task in zip(datasets, tasks):
            predictions = load_output_file(os.path.join(args.save_dir, "results", "{}.txt".format(dataset)))
            labels = [dp["label_list"][dp["label"]] for dp in task.examples]
            acc = np.mean(np.array(predictions)==np.array(labels))
            row.append("%.1f" % (100*acc))
        pt.add_row(row)

        for model in ["npm-single", "npm"]:
            row = [model]
            model_dir = os.path.join(args.save_dir, model, "results")
            for dataset, task in zip(datasets, tasks):
                output_files = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if file.startswith(dataset+"_c=")]
                if len(output_files)>0 and os.path.exists(output_files[0]):
                    assert len(output_files)==1, output_files
                    predictions = load_output_file(output_files[0])
                    predictions = [p["prediction"] for p in predictions]
                    labels = [dp["label_list"][dp["label"]] for dp in task.examples]
                    assert len(predictions)==len(labels)
                    acc = np.mean(np.array(predictions)==np.array(labels))
                    row.append("%.1f" % (100*acc))
                else:
                    row.append("-")
            pt.add_row(row)

        print (pt)

    if args.open:

        datasets = ["lama-trex", "lama-google_re", "kamel", "triviaqa", "nq"]
        field_names = ["trex", "trex uhn", "trex hard", "gre", "gre uhn", "kml", "tqa", "nq"]

        tasks = []
        for dataset in datasets:
            tasks.append(Task(dataset, "data", n_samples=3000))

        pt = PrettyTable()
        pt.field_names = ["Model"] + field_names
        pt.align["Model"] = "l"
        for dataset in datasets:
            pt.align[dataset] = "r"

        def compute_macro_em(accs, task, filter_func=None):
            acc_dict = defaultdict(list)
            for acc, ex, ngram in zip(accs, task.examples, task.ngrams):
                if filter_func is not None and not filter_func(ex):
                    continue
                acc_dict[ngram].append(acc)
            return np.mean([np.mean(_accs) for _accs in acc_dict.values()])

        def get_row(dataset, output_file):
            predictions = load_output_file(output_file)
            predictions = [p["a=0.0"] if "a=0.0" in p else p["prediction"] for p in predictions]
            predictions = [normalize_answer(p) for p in predictions]
            references = [[normalize_answer(a) for a in ex["answers"]] for ex in task.examples]
            assert len(predictions)==len(references)
            accs = [prediction in reference for prediction, reference in zip(predictions, references)]
            row = []
            if dataset in ["lama-trex", "lama-google_re"]:
                row.append("%.1f" % (100*compute_macro_em(accs, task)))
                row.append("%.1f" % (100*compute_macro_em(accs, task, lambda x: x["is_uhn"])))
                if dataset=="lama-trex":
                    row.append("%.1f" % (100*compute_macro_em(accs, task, lambda x: x["is_hard"])))
            else:
                row.append("%.1f" % (100*np.mean(accs)))
            return row

        for model_name in os.listdir(args.save_dir):
            if model_name.startswith("opt") or model_name.startswith("neo") or model_name.startswith("gpt"):
                row = [model_name]
                for dataset, task in zip(datasets, tasks):
                    output_file = os.path.join(args.save_dir, model_name, "{}.jsonl".format(output_file))
                    if os.path.exists(output_file):
                        row += get_row(dataset, output_file)
                    else:
                        if dataset=="lama-trex":
                            n = 3
                        elif dataset=="lama-google_re":
                            n = 2
                        else:
                            n = 1
                        for _ in range(n):
                            row.append("-")
                pt.add_row(row)

        row = ["npm"]
        model_dir = os.path.join(args.save_dir, "npm-reproduced", "results")
        for dataset, task in zip(datasets, tasks):
            output_files = [os.path.join(model_dir, file)
                            for file in os.listdir(model_dir)
                            if file.startswith(dataset+"_c=enwiki:no_stopwords_")]
            if len(output_files)>0 and os.path.exists(output_files[0]):
                assert len(output_files)==1, output_files
                row += get_row(dataset, output_files[0])
            else:
                if dataset=="lama-trex":
                    n = 3
                elif dataset=="lama-google_re":
                    n = 2
                else:
                    n = 1
                for _ in range(n):
                    row.append("-")

        pt.add_row(row)
        print (pt)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="save")
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    assert args.closed or args.open

    main(args)


