import argparse
import json
import numpy as np
from collections import defaultdict


def make_translation_context_dataset(path, src1, src2, tgt, split_tok=128022):

    def load_data(src, tgt):
        src_size = json.load(
            open(f"{model_path}/data-knnds-{src}_{tgt}/config.json", "r")
        )["data_infos"]["vals"]["shape"][0]
        
        # src_size=2000

        vals = [int(x) for x in np.memmap(f"{model_path}/data-knnds-{src}_{tgt}/vals.npy", dtype=int, mode="r", shape=(src_size, 1))]
        keys = np.memmap(f"{model_path}/data-knnds-{src}_{tgt}/keys.npy", dtype=np.float16, mode="r", shape=(src_size, 1024))

        current = []
        sentences, positions = [], []
        sentinfo = defaultdict(list)

        for idx, val in enumerate(vals):
            if val == split_tok:
                if current:
                    sent = " ".join([str(c) for c in current])
                    pos = [idx-len(current), idx-1]
                    sentences.append(sent)
                    positions.append(pos)
                    sentinfo[sent].append(pos)


                current = []
                current.append(val)
            else:
                current.append(val)

        return sentinfo, keys, vals

    print("Loading data...")
    src1_tgt_sents, src1_keys, src1_vals = load_data(src1, tgt)
    print("Loaded src1")
    src2_tgt_sents, src2_keys, src2_vals = load_data(src2, tgt)
    print("Loaded src2")

    print("Source sents: ",len(src1_tgt_sents), "Tgt sents: ",len(src2_tgt_sents))

    matches = 0
    src1_pos_to_src2_pos = defaultdict(list)
    for src1_tgt_sent in src1_tgt_sents.keys():
        if len(src2_tgt_sents[src1_tgt_sent]) > 0:
            matches +=1

            for src1_pos in src1_tgt_sents[src1_tgt_sent]:
                src1_pos = [i for i in range(src1_pos[0], src1_pos[1])]

            for src2_pos in src2_tgt_sents[src1_tgt_sent]:
                src2_pos = [i for i in range(src2_pos[0], src2_pos[1])]

            src1_pos = [[i for i in range(x[0], x[1])] for x in src1_tgt_sents[src1_tgt_sent]]
            src2_pos = [[i for i in range(x[0], x[1])] for x in src2_tgt_sents[src1_tgt_sent]]

            for idx_sent1, pos_sent1 in enumerate(src1_pos):
                for idx_val1, pos_val1 in enumerate(pos_sent1):
                        src1_pos_to_src2_pos[pos_val1].extend([val2[idx_val1] for val2 in src2_pos])

    # Check if no mistakes
    for pos in src1_pos_to_src2_pos.keys():
        # should only be 1 value; always the same target token
        assert len(set([src2_vals[x] for x in src1_pos_to_src2_pos[pos]])) == 1
        # should be same for src1 and src2
        assert src1_vals[pos] == [src2_vals[x] for x in src1_pos_to_src2_pos[pos]][0]

    print("#Matching sents:  ",matches)
    print("#Matching tokens: ",sum([len(x) for x in src1_pos_to_src2_pos.values()]))

    print("Creating training data...")
    np.random.seed(42)
    X, Y = [], []
    for src1_pos in src1_pos_to_src2_pos.keys():
        for src2_pos in src1_pos_to_src2_pos[src1_pos]:
            # print(src1_pos, src2_pos)

            X.append(src1_keys[src1_pos].astype(np.float32))
            Y.append(src2_keys[src2_pos].astype(np.float32))


    X, Y = np.array(X), np.array(Y)
    print("Data size: ", len(X))

    print("Shuffling data...")
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]

    # Split into train valid test
    length = len(X)
    test_end = int(length * 0.9)
    X_train, Y_train = X[:test_end], Y[:test_end]
    X_test, Y_test = X[test_end:], Y[test_end:]



    print("MAD(X_test, Y_test) = ", np.mean(np.abs(X_test-Y_test)))

    # for filename, file in zip(
    #     ["X_train.npy", "Y_train.npy", "X_test.npy", "Y_test.npy"],
    #     [X_train, Y_train, X_test, Y_test]
    # ):
    #     np.save(f"{model_path}/{filename}", file)

    print("Calculating least squares")
    W, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)

    # X@W ~= Y
    print("LSQ MAD(X_test, Y_test) = ", np.mean(np.abs((X_test@W)-Y_test)))

    print("Saving least squares result...")
    np.save(f"{model_path}/data-knnds-{src2}_{tgt}/map_{src1}_to_{src2}.npy", W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Mowgli NMT")
    ap.add_argument("--src1", type=str)
    ap.add_argument("--src2", type=str)

    args = ap.parse_args()

    model_path = "/ivi/ilps/personal/dstap1/data/fairseq_ted_m2m_100"
    tgt = "en"

    make_translation_context_dataset(path=model_path, src1=args.src1, src2=args.src2, tgt=tgt)