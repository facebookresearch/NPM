# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import time
import numpy as np
import json
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from utils_span import mask_spans

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="train_corpus")
    parser.add_argument("--mr", type=float, default=0.15)
    parser.add_argument("--p", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--num_shards", type=int, default=10)

    args = parser.parse_args()

    if args.analysis:
        analysis(args)
        return

    ext = "_mr{}_p{}.jsonl".format(args.mr, args.p)

    def find_files(out_dir):
        if os.path.isdir(out_dir):
            return sorted([fn for sub_dir in os.listdir(out_dir) for fn in find_files(os.path.join(out_dir, sub_dir))])

        fn = out_dir
        if fn.split("/")[-1].startswith("BS{}_shard".format(args.batch_size)) and fn.endswith(".jsonl"):
            if fn.endswith(ext):
                return []
            if os.path.exists(fn.replace(".jsonl", ext)):
                return []
            return [fn]

        return []

    filenames = find_files(args.data_dir)
    filenames = [fn for fn in filenames if fn.split(".")[-2][-3] not in ["6", "7", "8"]]
    print ("Start span masking for %d files" % len(filenames))
    f = partial(mask_spans,
                masking_ratio=args.mr,
                p=args.p)

    tot = 0
    with Pool(min(len(filenames), 80)) as p:
        for _ in p.imap(f, filenames):
            tot += 1


def analysis(args):
    import json
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    mask_id = tokenizer.mask_token_id

    def load(fn):
        print ("Starting loading", fn)
        data = []
        raw_text_to_position = {}
        with open(fn, "r") as f:
            for line in f:
                dp = json.loads(line)
                for i, raw_text in enumerate(dp["contents"]):
                    raw_text_to_position[raw_text] = (len(data), i)
                data.append(dp)
                if len(data)==3000:
                    break
        return data, raw_text_to_position

    def backgrounded(text, color):
        return "<span style='background-color: {}'>{}</span>".format(color, text)

    def decode(masked_input_ids_list, merged_labels):
        decoded_list = []
        colors = ["#FAF884", "#E2FAB5"]
        for i, (labels, masked_input_ids) in enumerate(zip(merged_labels, masked_input_ids_list)):
            while masked_input_ids[-1]==0:
                masked_input_ids = masked_input_ids[:-1]
            decoded = tokenizer.decode(masked_input_ids)
            color_idx = 0
            for label in labels:
                assert "<mask>"*len(label) in decoded
                decoded = decoded.replace("<mask>"*len(label),
                                          backgrounded(tokenizer.decode(label), colors[color_idx]),
                                          1)
                color_idx = 1-color_idx
            assert "<mask>" not in decoded
            decoded_list.append(decoded.replace("<s>", "").replace("</s>", ""))
        return decoded_list

    if args.wiki:
        data_dir = "/private/home/sewonmin/data/enwiki/enwiki_roberta_tokenized"
        prefix = "enwiki0_grouped"
    else:
        data_dir = "/private/home/sewonmin/data/cc_news_en/cc_news_roberta_tokenized"
        prefix = "batch0" #_grouped_v4"

    output_file = os.path.join(data_dir, "{}_{}.jsonl".format(prefix, "mr0.4_p0.2"))
    output2_file = os.path.join(data_dir, "{}_{}_token_ids.jsonl".format(prefix, "mr0.4_p0.2"))
    output3_file = os.path.join(data_dir, "{}_{}.jsonl".format(prefix, "mr0.15_p0.5"))
    output4_file = os.path.join(data_dir, "{}_{}_token_ids.jsonl".format(prefix, "mr0.15_p0.5"))
    output5_file = os.path.join(data_dir, "{}_{}.jsonl".format(prefix, "mr0.15_p0.2"))
    output6_file = os.path.join(data_dir, "{}_{}_token_ids.jsonl".format(prefix, "mr0.15_p0.2"))

    if not os.path.exists(output_file):
        output_file = output_file.replace("batch", "")
    if not os.path.exists(output2_file):
        output2_file = output2_file.replace("batch", "")
    if not os.path.exists(output3_file):
        output3_file = output3_file.replace("batch", "")
    if not os.path.exists(output4_file):
        output4_file = output4_file.replace("batch", "")
    if not os.path.exists(output5_file):
        output5_file = output5_file.replace("batch", "")
    if not os.path.exists(output6_file):
        output6_file = output6_file.replace("batch", "")

    '''
    output_file = os.path.join(data_dir, "{}_{}.jsonl".format(0, "mr0.4_p0.2"))
    output2_file = os.path.join(data_dir, "{}_{}.jsonl".format(0, "mr0.4_p0.2"))
    output3_file = os.path.join(data_dir, "{}_{}_token_ids.jsonl".format(0, "mr0.4_p0.2"))
    output4_file = os.path.join(data_dir, "{}_{}_inv_token_ids.jsonl".format(0, "mr0.4_p0.2"))
    output5_file = os.path.join(data_dir, "{}_{}_token_ids_ent.jsonl".format(0, "mr0.15_p0.2"))
    output6_file = os.path.join(data_dir, "{}_{}_inv_token_ids_ent.jsonl".format(0, "mr0.4_p0.2"))
    '''

    start_time = time.time()
    np.random.seed(2022)
    data1, raw_text_to_position1 = load(output_file)
    data2, raw_text_to_position2 = load(output2_file)
    data3, raw_text_to_position3 = load(output3_file)
    data4, raw_text_to_position4 = load(output4_file)
    data5, raw_text_to_position5 = load(output5_file)
    data6, raw_text_to_position6 = load(output6_file)

    is_same = []

    with open("{}samples.html".format("wiki_" if args.wiki else ""), "w") as f:

        for dp_idx in range(50):
            dp = data3[dp_idx]
            masked_texts = decode(dp["masked_input_ids"], dp["merged_labels"])
            raw_texts = dp["contents"]

            if np.all([raw_text not in raw_text_to_position1 for raw_text in raw_texts]):
                continue

            for masked_text3, raw_text in zip(masked_texts, raw_texts):
                if raw_text not in raw_text_to_position1:
                    continue
                if raw_text not in raw_text_to_position2:
                    continue
                if raw_text not in raw_text_to_position4:
                    continue
                if raw_text not in raw_text_to_position5:
                    continue
                if raw_text not in raw_text_to_position6:
                    continue

                p = raw_text_to_position1[raw_text]
                other_input_ids = data1[p[0]]["masked_input_ids"][p[1]]
                other_labels = data1[p[0]]["merged_labels"][p[1]]
                masked_text1 = decode([other_input_ids], [other_labels])[0]

                p = raw_text_to_position2[raw_text]
                other_input_ids = data2[p[0]]["masked_input_ids"][p[1]]
                other_labels = data2[p[0]]["merged_labels"][p[1]]
                masked_text2 = decode([other_input_ids], [other_labels])[0]

                p = raw_text_to_position4[raw_text]
                other_input_ids = data4[p[0]]["masked_input_ids"][p[1]]
                other_labels = data4[p[0]]["merged_labels"][p[1]]
                masked_text4 = decode([other_input_ids], [other_labels])[0]

                p = raw_text_to_position5[raw_text]
                other_input_ids = data5[p[0]]["masked_input_ids"][p[1]]
                other_labels = data5[p[0]]["merged_labels"][p[1]]
                masked_text5 = decode([other_input_ids], [other_labels])[0]

                p = raw_text_to_position6[raw_text]
                other_input_ids = data6[p[0]]["masked_input_ids"][p[1]]
                other_labels = data6[p[0]]["merged_labels"][p[1]]
                masked_text6 = decode([other_input_ids], [other_labels])[0]

                is_same.append(masked_text3==masked_text4)

                '''
                f.write("<strong>w/ BM25 (inv_token_ids, ent):</strong> {}<br /><br />".format(masked_text6))
                f.write("<strong>w/ BM25 (inv_token_ids):</strong> {}<br /><br />".format(masked_text5))
                f.write("<strong>w/ BM25 (token_ids, ent):</strong> {}<br /><br />".format(masked_text4))
                f.write("<strong>w/ BM25 (token_ids):</strong> {}<br /><br />".format(masked_text3))
                f.write("<strong>w/ BM25:</strong> {}<br /><br />".format(masked_text2))
                f.write("<strong>w/o BM25:</strong> {}<br /><br />".format(masked_text1))
                '''
                f.write("<strong>w/o BM25 (token_ids, 0.15, 0.2):</strong> {}<br /><br />".format(masked_text6))
                f.write("<strong>w/o BM25 (0.15, 0.2):</strong> {}<br /><br />".format(masked_text5))
                f.write("<strong>w/o BM25 (token_ids, 0.15, 0.5):</strong> {}<br /><br />".format(masked_text4))
                f.write("<strong>w/o BM25 (0.15, 0.5):</strong> {}<br /><br />".format(masked_text3))
                f.write("<strong>w/o BM25 (token_ids, 0.4, 0.2):</strong> {}<br /><br />".format(masked_text2))
                f.write("<strong>w/o BM25 (0.4, 0.2):</strong> {}<br /><br />".format(masked_text1))


            f.write("<hr />")

    print (np.mean(is_same))

if __name__=='__main__':
    main()



