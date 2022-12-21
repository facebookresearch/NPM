# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import argparse
import numpy as np

from prettytable import PrettyTable
from task.task import Task

datasets = ["agn", "yahoo", "subj", "sst2", "mr", "rt", "cr", "amazon", "rte"]

tasks = []
for dataset in datasets:
    tasks.append(Task(dataset, "data", n_samples=3000))

def load_output_file(output_file):
    predictions = []
    with open(output_file, "r") as f:
        for line in f:
            predictions.append(json.loads(line)["prediction"])
    return predictions

def main(args):

    if args.closed:
        pt = PrettyTable()
        pt.field_names = ["Model"] + datasets
        pt.align["Model"] = "l"
        for dataset in datasets:
            pt.align[dataset] = "r"

        row = ["RoBERTa"]
        for dataset, task in zip(datasets, tasks):
            predictions = load_output_file("save/results/{}.txt".format(dataset))
            labels = [dp["label_list"][dp["label"]] for dp in task.examples]
            acc = np.mean(np.array(predictions)==np.array(labels))
            row.append("%.1f" % (100*acc))
        pt.add_row(row)

        for model in ["npm", "npminf"]:
            row = [model]
            model_dir = "/checkpoint/sewonmin/npm_checkpoints/{}/results".format(model)
            for dataset, task in zip(datasets, tasks):
                output_files = [os.path.join(model_dir, file) for file in os.listdir(model_dir) if file.startswith(dataset+"_c=")]
                if len(output_files)>0 and os.path.exists(output_files[0]):
                    assert len(output_files)==1, output_files
                    predictions = load_output_file(output_files[0])
                    labels = [dp["label_list"][dp["label"]] for dp in task.examples]
                    acc = np.mean(np.array(predictions)==np.array(labels))
                    row.append("%.1f" % (100*acc))
                else:
                    row.append("-")
            pt.add_row(row)

        print (pt)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    assert args.closed or args.open

    main(args)


