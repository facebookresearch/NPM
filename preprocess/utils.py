import json
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
mask_id = tokenizer.mask_token_id

def create_blocks_from_plain_text(sentences, doc_idx, max_seq_length=256):
    input_ids = tokenizer._batch_encode_plus(sentences)["input_ids"]
    assert type(input_ids)==list

    curr_input_ids_list = [[]]
    for tokens in input_ids:

        if mask_id in tokens:
            # sometimes, the raw text contains [MASK]. in this case, we skip.
            continue

        if len(tokens) + len(curr_input_ids_list[-1]) <= max_seq_length:
            curr_input_ids_list[-1] += tokens
        elif len(tokens) <= max_seq_length:
            curr_input_ids_list.append(tokens)
        else:
            while len(tokens) > max_seq_length:
                th = max_seq_length-len(curr_input_ids_list[-1])
                curr_input_ids_list[-1] += tokens[:th]
                tokens = tokens[th:]
                curr_input_ids_list.append([])
            if len(tokens)>0:
                curr_input_ids_list[-1] += tokens

    output_lines = []
    n_tokens = []
    for block_idx, _input_ids in enumerate(curr_input_ids_list):
        assert 0<len(_input_ids)<=max_seq_length
        n_tokens.append(len(_input_ids))
        raw_text = tokenizer.decode(_input_ids)
        padded_input_ids = _input_ids + [0 for _ in range(max_seq_length-len(_input_ids))]
        attention_mask = [1 for _ in _input_ids] + [0 for _ in range(max_seq_length-len(_input_ids))]
        assert len(padded_input_ids)==len(attention_mask)==max_seq_length
        out_dp = {"id": "{}-{}".format(doc_idx, block_idx),
                  "block_idx": block_idx,
                  "contents": raw_text,
                  "input_ids": padded_input_ids,
                  "attention_mask": attention_mask}
        output_lines.append(out_dp)

    return output_lines, n_tokens


