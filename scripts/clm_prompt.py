# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, AutoTokenizer
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast

from task.task import Task
from scripts.util_clm import convert_model_to_int8_on_gpu

class Model(object):

    def __init__(self, model_name="neo-1.3b"):
        print ("Start loading %s" % model_name)
        start_time = time.time()
        if model_name=="j-6b":
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        elif model_name=="neo-1.3b":
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        elif model_name=="neo-2.7b":
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        elif model_name=="neox-20b":
            self.model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
            self.tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        elif model_name.startswith("opt-"):
            assert model_name in ["opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b", "opt-30b"]
            self.model = AutoModelForCausalLM.from_pretrained("facebook/" + model_name, torch_dtype=torch.float16)
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/" + model_name)
        elif model_name.startswith("bloom"):
            assert model_name in ["bloom-1b7", "bloom-3b", "bloom-7b1"]
            self.model = BloomForCausalLM.from_pretrained("bigscience/" + model_name, torch_dtype=torch.float16)
            self.tokenizer = BloomTokenizerFast.from_pretrained("bigscience/" + model_name)
        else:
            raise NotImplementedError()
        self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        print ("Takes %ds to load" % (time.time()-start_time))
        print ("\t%.1fB parameters" % (
            np.sum([p.numel() for p in self.model.parameters()])/1000000000))

    def generate(self, prompts, max_sequence_length=2048, max_output_length=30):
        input_ids = self.tokenizer(prompts).input_ids

        generations = []
        for curr_input_ids in tqdm(input_ids):
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            gen_tokens = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
            )
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:]).split("\n")[0].strip()

            if len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            generations.append(gen)

        assert len(generations)==len(prompts)
        return generations

import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load_inputs(args):
    ret = args.ret
    assert ret is None or ret in ["bm25", "bm25_2022"]

    task = Task(args.eval_dataset, "data", n_samples=3000) #args.n_samples)

    if args.ret:
        from npm.searcher import BM25Searcher
        base_dir = "corpus"
        name = "new-enwiki" if ret=="bm25_2022" else "enwiki"
        data_dir = os.path.join(base_dir, name)
        index_dir = os.path.join(base_dir, name + "-index")
        searcher = BM25Searcher(data_dir, index_dir)
        restricted, restricted_dict = searcher.batch_search(task)

        text_dict = {}
        offset = 0
        for i in range(20):
            with open(os.path.join(data_dir, "{}.jsonl".format(i)), "r") as f:
                for line in f:
                    if offset in restricted:
                        text_dict[offset] = json.loads(line)
                    offset += 1
        assert len(text_dict)==len(restricted)


    inputs = []
    outputs = []
    ngrams = []

    for ex in task.examples:
        # use a slightly different template that is better for CLMs
        if args.eval_dataset in ["nq", "triviaqa", "kamel"]:
            question = ex["input"].split("The answer is: <mask>")[0].strip()
            input_text = "Question: {}\nAnswer:".format(question)
        elif args.eval_dataset=="entity_translation":
            input_text = ex["input"].split("<mask>")[0].strip()
        elif args.eval_dataset.startswith("lama-"):
            input_text = ex["input"].replace("<mask>", "_____").strip() + " Fill in the blank. Answer:"
        else:
            raise NotImplementedError()

        if ret:
            for p in restricted_dict[ex["input"]]:
                p = text_dict[p]
                input_text = p["contents"].replace("<s>", "").replace("</s>", "").strip() + "\n" + input_text

        inputs.append(input_text)
        outputs.append(ex["answers"])

    print ("Input:", inputs[0])
    print ("Output:", outputs[0])

    return task, inputs, outputs


def calc_accuracy(task, outputs, predictions):

    def postprocess_prediction(text):
        if str(task)=="entity_translation":
            # this seems to be helpful in entity translation
            for i in range(len(text)):
                if text[i] in string.punctuation:
                    text = text[:i]
                    break
        return text

    references = [[normalize_answer(answer) for answer in output] for output in outputs]
    predictions = [normalize_answer(postprocess_prediction(p)) for p in predictions]
    accs = [prediction in reference for prediction, reference in zip(predictions, references)]

    if task.ngrams is not None:
        accs_dict = defaultdict(list)
        for acc, ngram in zip(accs, task.ngrams):
            accs_dict[ngram].append(acc)
        acc = np.mean([np.mean(v) for k, v in accs_dict.items()])
        print ("\tMacro EM=%.1f%%" % (100*acc))
    else:
        acc = np.mean(accs)
        print ("\tEM=%.1f%%" % (100*acc))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="neo-1.3B")
    parser.add_argument("--eval_dataset",
                        type=str,
                        default=None)
    parser.add_argument("--save_dir",
                        type=str,
                        default="save")
    parser.add_argument("--ret", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    task, inputs, outputs = load_inputs(args)
    print ("Took %dsec to load the data" % (time.time()-start_time))

    prediction_path = os.path.join(args.save_dir,
                                   args.model_name,
                                   "{}{}.jsonl".format(args.eval_dataset,
                                                         "" if args.ret is None else "_" + args.ret)
                                   )

    if not args.eval_only:
        model = Model(args.model_name)
        start_time = time.time()
        predictions = model.generate(inputs, max_sequence_length=args.max_sequence_length)
        print ("Took %dsec to generate" % (time.time()-start_time))

        if not os.path.exists(os.path.join(args.save_dir, args.model_name)):
            os.makedirs(os.path.join(args.save_dir, args.model_name))

        with open(prediction_path, "w") as f:
            for prediction in predictions:
                f.write(json.dumps({"prediction": prediction}) + "\n")
    else:
        assert os.path.exists(prediction_path)
        with open(prediction_path, "r") as f:
            predictions = []
            for line in f:
                predictions.append(json.loads(line)["prediction"])

    calc_accuracy(task, outputs, predictions)


if __name__=='__main__':
    main()
