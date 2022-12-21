# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import csv
import json
import string

from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

sentiment_analysis_labels = [" terrible", " great"]
yahoo_topics = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]

topic_classification_premise = "The text topic is about"

def load_data(dataname, data_dir):
    ## open-set tasks
    if dataname.startswith("lama-"):
        return load_lama(data_dir, subset=dataname[5:])
    elif dataname=="triviaqa":
        return load_triviaqa()
    elif dataname=="nq":
        return load_nq(data_dir)
    elif dataname=="kamel":
        return load_kamel(data_dir)

    ## closed-set tasks

    if dataname=="yahoo":
        data = load_yahoo()
    elif dataname=="amazon":
        data = load_amazon()
    elif dataname=="rte":
        data = load_rte()
    elif dataname=="rt":
        data = load_rt()
    else:
        data_path = os.path.join(data_dir, dataname)
        data_files = os.listdir(data_path)
        test_data_files = [fn for fn in data_files if fn.startswith("test")]
        if len(test_data_files)==0:
            test_data_files = [fn for fn in data_files if fn.startswith("dev")]
        assert len(test_data_files)==1, (dataname, test_data_files)
        test_data_file = os.path.join(data_dir, dataname, test_data_files[0])

        data = []
        if dataname=="agn":
            data = load_agn(test_data_file)
        elif dataname=="sst2":
            data = load_sst2(test_data_file)
        elif test_data_file.endswith(".csv") or test_data_file.endswith(".tsv"):
            with open(test_data_file) as f:
                reader = csv.DictReader(f, fieldnames=["label", "text"])
                for i, row in enumerate(reader):
                    data.append({"input": row["text"], "label": int(row["label"])})
        elif test_data_file.endswith(".jsonl"):
            with open(test_data_file, "r") as f:
                for line in f:
                    dp = json.loads(line)
                    data.append(dp)
        else:
            raise NotImplementedError(test_data_file)

    if dataname in ["sst2", "mr", "rt", "cr", "subj"]:
        for i, dp in enumerate(data):
            data[i]["input"] = detokenize(dp["input"])

    if dataname in ["sst2", "mr", "rt", "cr", "amazon"]:
        for i, dp in enumerate(data):
            data[i]["input"] = dp["input"].strip() + " It was"
            data[i]["label_list"] = sentiment_analysis_labels

    elif dataname=="subj":
        for i, dp in enumerate(data):
            data[i]["input"] = dp["input"].strip() + " This is a"
            data[i]["label_list"] = [" review", " summary"]

    return data

def load_fuzzy_verbalizer(path):
    if path.endswith(".txt"):
        label2syn = defaultdict(list)
        with open(path) as f:
            for i, line in enumerate(f):
                for w in line.strip().split(", "):
                    label2syn[i].append(f" {w}")
        min_len = min([len(v) for v in label2syn.values()])
        for k, v in label2syn.copy().items():
            label2syn[k] = v[:min_len]
        return label2syn
    elif path.endswith("yahoo.json"):
        label2syn = {}
        with open(path, "r") as f:
            topic2syn = json.load(f)
        for i, (k, v) in enumerate(topic2syn.items()):
            label2syn[yahoo_topics.index(k)] = [" "+item for item in v]
    else:
        raise NotImplementedError(path)
    return label2syn

def detokenize(text):
    punc = set(["!", "?", ".", ",", "'", ";", ":", ")", "]"])

    text = text.replace("n 't", "n't")
    tokens = text.split()
    text = ""
    no_space_next_token = False
    for i, token in enumerate(tokens):
        if i==0:
            pass
        elif no_space_next_token:
            no_space_next_token = False
        elif token in punc or \
                (len(token)>=2 and token.startswith("'")) or \
                (len(token)>=3 and token=="n't"):
            pass
        elif text.endswith("www."):
            pass
        elif text.endswith(".") and token=="com":
            pass
        elif token in ["(", "["]:
            token = " " + token
            no_space_next_token = True
        else:
            token = " " + token
        text += token
    return text.replace("!.", "!").replace("?.", "?")

def load_sst2(data_file):
    data = []
    with open(data_file) as f:
        for line in f:
            l, s = line.strip().split('\t')
            label = int(l[-1])-3
            if label == 0:
                continue
            label = 1 if label > 0 else 0
            data.append({"input": s, "label": label})
    return data

def load_agn(data_file):
    topics = [' politics', ' sports', ' business', ' technology']
    examples = []
    with open(data_file) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['Class Index'])-1
            title = row['Title']
            summary = row['Description']
            input_text = f"{title} \n {summary} {topic_classification_premise}"
            examples.append({'label' : label, 'label_list': topics, 'input': input_text})
    return examples

def load_rt():
    data = load_dataset("rotten_tomatoes", split="test")
    examples = []
    for dp in data:
        examples.append({"input": dp["text"], "label": dp["label"]})
    return examples

def load_yahoo():
    label_list = [" society", " science", " health", " education", " computer", " sports", " business", " entertainment", " family", " politics"]
    data = load_dataset("yahoo_answers_topics", split="test")

    prompt = " " + topic_classification_premise
    icl_str = ""

    examples = []
    for row in data:
        label = row["topic"]
        title = row['question_title']
        summary = row['question_content']
        answer = row['best_answer']
        input_text = f"title: {title} content: {summary} answer: {answer}{prompt}"
        examples.append({'input': input_text, 'label' : label, 'label_list': label_list})
    return examples

def load_amazon():
    data = load_dataset("amazon_polarity", split="test")

    def preprocess(title, content):
        if not any([title.endswith(p) for p in string.punctuation]):
            title = title + "."
        return title + " " + content

    examples = []
    for dp in data:
        examples.append({"input": preprocess(dp["title"], dp["content"]),
                         "label": dp["label"]})
    return examples

def load_rte():
    data = load_dataset("glue", "rte", split="validation") #, verbose=False)
    puncutations = set(string.punctuation)
    label_list = [" Yes", " No"]

    examples = []
    for i, dp in enumerate(data):
        pre = dp["sentence1"]
        hyp = dp["sentence2"]
        label = dp["label"]
        while pre[-1] in puncutations:
            pre = pre[:-1]
        while hyp[-1] in puncutations:
            hyp = hyp[:-1]
        input_text = pre + ", right?<mask>, " + hyp[0].lower() + hyp[1:] + "."
        examples.append({"input": input_text, "label": label, "label_list": label_list})
    return examples

def template_question(question):
    question = question.strip()
    if question.startswith('"') and question.endswith('"'):
        question = question[1:-1].strip()
    if not question.endswith("?"):
        question = question + "?"
    return question + " The answer is: <mask>."

def load_triviaqa():
    orig_data = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
    data = []
    for dp in orig_data:
        answers = [dp["answer"]["value"]]
        for alias in dp["answer"]["aliases"] + dp["answer"]["normalized_aliases"]:
            answers.append(alias)
        data.append({"input": template_question(dp["question"]),
                     "answers": answers})
    return data

def load_nq(data_dir, split="test"):
    with open(os.path.join(data_dir, "nq/nqopen-{}.json".format(split)), "r") as f:
        orig_data = json.load(f)
    with open(os.path.join(data_dir, "nq/{}_id2answers.json".format(split)), "r") as f:
        id2answers = json.load(f)
    data = []
    for dp in orig_data:
        data.append({"input": template_question(dp["question"]),
                     "answers": id2answers[dp["id"]]})
    return data

def load_lama(data_dir, subset):

    data_path = os.path.join(data_dir, "lama", "{}.jsonl".format(subset))

    if os.path.exists(data_path):
        data = []
        with open(data_path, "r") as f:
            for line in f:
                data.append(json.loads(line))

    else:
        base_dir = os.path.join(data_dir, "lama", "data")
        id2hf_data = _load_hf_lama(subset)

        if subset=="trex":
            data = _load_lama(os.path.join(base_dir, "LAMA-TREx"), "trex", id2hf_data)
            data1 = _load_lama(os.path.join(base_dir, "LAMA-TREx_UHN"), "trex", id2hf_data)
            data2 = _load_lama(os.path.join(base_dir, "LAMA-TREx-easy-hard/Hard"), "trex", id2hf_data)

            ids = set([dp["id"] for dp in data])
            ids1 = set([dp["id"] for dp in data1])
            ids2 = set([dp["id"] for dp in data2])
            assert len(ids1-ids)==len(ids2-ids)==0

            for i, dp in enumerate(data):
                data[i]["is_uhn"] = dp["id"] in ids1
                data[i]["is_hard"] = dp["id"] in ids2

        elif subset=="google_re":
            data = _load_lama(os.path.join(base_dir, "Google_RE"), "google_re", id2hf_data)
            data1 = _load_lama(os.path.join(base_dir, "Google_RE_UHN"), "google_re", id2hf_data)
            ids = set([dp["id"] for dp in data])
            ids1 = set([dp["id"] for dp in data1])
            assert len(ids1-ids)==0

            for i, dp in enumerate(data):
                data[i]["is_uhn"] = dp["id"] in ids1

        # we need to tokenize answers since it is needed for computing macro-average and sampling the data
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        def tokenize(answer):
            if type(answer)==list:
                return [tokenize(_answer) for _answer in answer]
            input_ids = tokenizer(" " + answer.strip())["input_ids"]
            assert input_ids[0]==0 and input_ids[-1]==2
            return input_ids[1:-1]

        for i, dp in enumerate(data):
            answers = dp["answers"]
            data[i]["tokenized_answers"] = tokenize(answers)

        with open(data_path, "w") as f:
            for dp in data:
                f.write(json.dumps(dp)+"\n")

    return data

def _load_hf_lama(name):

    ## Load HF version first to use their template (in case of Google RE) and double-check the data matches

    def convert_input(dp):
        text = dp["masked_sentence"]
        inputs = set()
        if "template" in dp:
            text = dp["template"]
            text = text.replace("[Y]", "<mask>").replace(" .", ".")
            text = text.replace("[X]", dp["sub_label"])
            inputs.add(text)
        return inputs

    dataset = load_dataset("lama", name, split="train")
    id2data = defaultdict(list)
    for dp in tqdm(dataset):
        sub = dp["sub_label"] if "sub_label" in dp else dp["sub"]
        obj = dp["obj_label"]
        inputs = convert_input(dp)

        if dp["masked_sentence"].count("[MASK]")!=1:
            # print ("Skipping `%s` from %s" % (dp["masked_sentence"], name))
            continue

        assert all([input_.count("<mask>")==1 for input_ in inputs])

        if dp["uuid"] in id2data:
            old_dp = id2data[dp["uuid"]]
            assert old_dp["subject"]==sub and old_dp["object"]==obj
            id2data[dp["uuid"]]["inputs"] |= inputs
        else:
            id2data[dp["uuid"]] = {"inputs": inputs, "subject": sub, "object": obj}

    return id2data

def _load_lama(data_dir, name, id2hf_data):
    data = []
    for relation in os.listdir(data_dir):
        with open(os.path.join(data_dir, relation), "r") as f:
            for line in f:
                dp = json.loads(line)
                subjects = set()
                objects = set()
                inputs = set()

                subjects.add(dp["sub_label"])
                objects.add(dp["obj_label"])

                if "masked_sentences" in dp:
                    text = " ".join(dp["masked_sentences"])
                    if text.count("[MASK]")!=1:
                        continue
                    text = text.replace("[MASK]", "<mask>").replace(" .", ".")
                    inputs.add(text)

                for e in dp["evidences"]:
                    if "sub_surface" in e:
                        subjects.add(e["sub_surface"])
                    if "obj_surface" in e:
                        objects.add(e["obj_surface"])

                assert dp["uuid"] in id2hf_data
                if name!="google_re":
                    if dp["uuid"] in id2hf_data:
                        hf_dp = id2hf_data[dp["uuid"]]
                        subjects.add(hf_dp["subject"])
                        objects.add(hf_dp["object"])
                        inputs |= hf_dp["inputs"]

                if len(inputs)==0:
                    continue

                if name=="google_re":
                    # weirdly, the template from the original data is more like evidence from
                    # original sources, so we are using templates from HF
                    inputs = id2hf_data[dp["uuid"]]["inputs"]

                if len(inputs)==0:
                    continue

                inputs = list(inputs)
                assert len(inputs)==1
                assert inputs[0].count("<mask>")==1
                data.append({
                    "id": dp["uuid"],
                    "subjects": list(subjects),
                    "answers": list(objects),
                    "input": inputs[0]
                })
    return data

def load_kamel(data_dir):
    data_dir = os.path.join(data_dir, "kamel")
    template_path = os.path.join(data_dir, "question-templates.csv")

    import csv
    rel2question = {}
    with open(template_path, "r") as f:
        for rel, question in csv.reader(f):
            assert "[S]" in question, question
            assert question.endswith("?"), question
            rel2question[rel] = question

    data = []
    for rel in tqdm(os.listdir(data_dir)):
        if not rel.startswith("P"):
            continue
        with open(os.path.join(data_dir, rel, "dev.jsonl"), "r") as f:
            for line in f:
                dp = json.loads(line)
                input_text = rel2question[rel].replace("[S]", dp["sub_label"]) + " Answer: <mask>."
                objects = dp["obj_label"]

                if any([o.endswith(".") or o.endswith("+") or o.endswith("!") for o in objects]):
                    input_text = input_text[:-1]

                data.append({"input": input_text, "answers": objects})

    return data




