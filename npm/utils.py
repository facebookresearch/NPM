import numpy as np
import time
import pickle as pkl
from collections import defaultdict

def load_sharded_data(data_path,
                      remove_stopwords,
                      stopwords):

    input_ids = np.load(data_path)
    input_ids_list = []
    token_idx_to_block_idx = []
    token_idx_to_local_idx = []
    orig_block_idx_to_emb_token_idx = {}
    orig_block_idx_to_emb_token_idx = defaultdict(list) #{}
    orig_block_idx_to_valid_start = {}
    orig_block_idx_to_valid_end = {}

    start_end_pairs = np.load(data_path.replace(".npy", "_blocks.npy"))
    with open(data_path.replace(".npy", "_valid.pkl"), "rb") as f:
        valid_candidates = pkl.load(f)

    offset = 0
    dstore_size = 0
    true_dstore_size = 0

    start_time = time.time()

    for block_idx, (valid_start, valid_end) in enumerate(valid_candidates):
        start = start_end_pairs[block_idx]
        end = start_end_pairs[block_idx+1] if block_idx<len(start_end_pairs)-1 else len(input_ids)
        valid_idxs = set(valid_start) | set(valid_end)

        curr_dstore_size = 0
        for i, curr_token in enumerate(input_ids[start:end]):
            if i not in valid_idxs or (remove_stopwords and curr_token in stopwords):
                continue
            curr_dstore_size += 1

            # TODO: everything except local_idx should be later modified to be global
            token_idx_to_block_idx.append(len(input_ids_list))
            token_idx_to_local_idx.append(i)
            orig_block_idx_to_emb_token_idx[offset].append(true_dstore_size+i)

        orig_block_idx_to_valid_start[offset] = valid_start
        orig_block_idx_to_valid_end[offset] = valid_end

        dstore_size += curr_dstore_size
        # global_dstore_size += curr_dstore_size
        true_dstore_size += curr_dstore_size
        # global_true_dstore_size += curr_dstore_size

        input_ids_list.append(input_ids[start:end])
        offset += 1

        if (block_idx+1) % 100000 == 0:
            print ("%dK blocks (%dsec)" % ((block_idx+1) / 1000, time.time()-start_time))

    return {"offset": offset,
            "dstore_size": dstore_size,
            "true_dstore_size": true_dstore_size,
            "input_ids": input_ids_list,
            "token_idx_to_block_idx": token_idx_to_block_idx,
            "token_idx_to_local_idx": token_idx_to_local_idx,
            "orig_block_idx_to_emb_token_idx": orig_block_idx_to_emb_token_idx,
            "orig_block_idx_to_emb_token_idx": orig_block_idx_to_emb_token_idx,
            "orig_block_idx_to_valid_start": orig_block_idx_to_valid_start,
            "orig_block_idx_to_valid_end": orig_block_idx_to_valid_end,
            }


