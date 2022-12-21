# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import time
import numpy as np
from string import punctuation
from collections import defaultdict, Counter

base_dir = "/checkpoint/sewonmin/data/preprocess_data/enwiki_roberta_tokenized"

punctuation = set(punctuation)
langs = ["Chinese", "Japanese", "Korean",
         "Hebrew", "Hindi", "Arabic",
         "Cyrillic", "Polish", "Czech",
         "Greek", "Turkish", "Malayalam",
         "Mongolian", "Russian", "Tamil", "Thai",
         ]

def main():
    #create()

    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    def validate(answer, text):
        return " ".join([str(a) for a in answer]) in " ".join([str(t) for t in text])

    output_dir = "task_data/translation2"
    n_tokens = defaultdict(list)
    new_counter = Counter()
    data = []
    examples = defaultdict(list)
    with open(os.path.join(output_dir, "dev.jsonl"), "r") as f:
        for line in f:
            dp = json.loads(line)
            answer = dp["objects"][0]
            text = dp["oracle_passages"][0]["text"]
            tokenized_text = tokenizer(answer)["input_ids"]

            if " " + answer not in answer:
                tokenized_answer = tokenizer(answer)["input_ids"]
            else:
                tokenized_answer = tokenizer(" " + answer)["input_ids"]
            assert validate(tokenized_answer, tokenized_text)
            n_tokens[dp["lang"]].append(len(tokenized_answer))
            dp["tokenized_objects"] = [tokenized_answer]
            data.append(dp)
            if len(tokenized_answer) <= 20:
                if len(examples[dp["lang"]]) < 5:
                    if text.count(dp["lang"]) <= 2:
                        example_text = "{}<br />{}<br />{}".format(
                            dp["input"].replace("<mask>", "_____"),
                            answer,
                            text.replace("<s>", "").replace("</s>", ""))
                        examples[dp["lang"]].append(example_text)
                new_counter[dp["lang"]] += 1

    with open("entity_translation_samples.html", "w") as f:
        for lang, samples in examples.items():
            f.write("Language: {}<br />".format(lang))
            for sample in samples:
                f.write(sample + "<br />")
            f.write("<hr />")

    return

    #with open(os.path.join(output_dir, "dev.jsonl"), "w") as f:
    #    for dp in data:
    #        f.write(json.dumps(dp)+"\n")

    for lang, ns in n_tokens.items():
        print ("lang:", lang)
        tot = len(ns)
        n_10 = np.sum([n>10 for n in ns])
        n_20 = np.sum([n>20 for n in ns])
        n_30 = np.sum([n>30 for n in ns])
        print ("\t%d/%d (%.1f%%) with >10 tokens" % (n_10, tot, 100*n_10/tot))
        print ("\t%d/%d (%.1f%%) with >20 tokens" % (n_20, tot, 100*n_20/tot))
        print ("\t%d/%d (%.1f%%) with >30 tokens" % (n_30, tot, 100*n_30/tot))

    for lang, n in sorted(new_counter.items(), key=lambda x: -x[1]):
        print (lang, n)

    print ("Total", np.sum([v for v in new_counter.values()]))
    print ("Total (sampled)", np.sum([min(1000, v) for v in new_counter.values()]))


def create():
    lang_counter = Counter()
    data = defaultdict(list)
    start_time = time.time()
    line_cnt = 0

    for shard_idx in range(20):
        data_path = os.path.join(base_dir, "en_head_train_v2_shard{}_flatten_pv.jsonl".format(shard_idx))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                if dp["block_idx"]==0:
                    for lang in langs:
                        if lang + ":" in dp["raw_text"]:
                            if lang + ":" in " ".join(dp["raw_text"].split()[:30]):
                                start = dp["raw_text"].index(lang+":") + len(lang) + 1
                                end = start
                                is_valid = True
                                while dp["raw_text"][end] not in punctuation:
                                    end += 1
                                    if len(dp["raw_text"])==end:
                                        is_valid = False
                                        break
                                if not is_valid:
                                    continue
                                translation = dp["raw_text"][start:end].strip()
                                if len(translation)>0:
                                    data[dp["title"]].append({
                                        "text": translation,
                                        "lang": lang,
                                        "oracle_passages": [{
                                            "offset": line_cnt,
                                            "text": dp["raw_text"]
                                        }]
                                    })
                                    lang_counter[lang] += 1
                line_cnt += 1
        print ("%d-th index finished!" % shard_idx)
        print ("Saved %d words and %d translations" % (
            len(data), np.sum([len(d) for d in data.values()])))
        print ("Took %dmin" % ((time.time()-start_time)/60))

    print (lang_counter)

    output_dir = "task_data/translation2"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, "dev.jsonl"), "w") as f:
        n_examples = 0
        for title, examples in data.items():
            for ex in examples:
                ex["question"] = "What is the {} translation of {}? The answer is: <mask>.".format(
                    ex["lang"], title
                )
                ex["input"] = "The {} translation of {} is: <mask>.".format(
                    ex["lang"], title
                )
                ex["objects"] = [ex["text"]]
                f.write(json.dumps(ex)+"\n")
                n_examples += 1
    print ("Finish saving %d examples at %s" % (n_examples, output_dir))


if __name__=='__main__':
    main()


