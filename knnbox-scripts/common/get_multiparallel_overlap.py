import os
from collections import defaultdict
import pickle
from fairseq.data import Dictionary, MMapIndexedDataset

data_path = "/ivi/ilps/personal/dstap1/data/fairseq_ted_m2m_100/data-bin" 
src_langs = ["da", "nl", "de", "sv", "fr", "gl", "it", "pt", "ro", "es", "be", "bs", "bg", "hr", "cs", "mk", "pl", "ru", "sr", "sk", "sl", "uk", "et", "fi", "hu", "lt", "sq", "hy", "ka", "el", "az", "kk", "tr", "ja", "ko", "vi", "zh", "bn", "hi", "ur", "ta", "id", "ms", "my", "th", "mn", "ar", "he"]

tgt_lang = "en"
dictionary = Dictionary.load(os.path.join(data_path, "dict.en.txt"))  # Replace with your actual dictionary path

datasets = {}

# Load datasets
for src_lang in src_langs:
    print(f"Loading {src_lang} data...")
    src_dataset_path = os.path.join(data_path, f"train.{src_lang}-{tgt_lang}.{src_lang}")
    tgt_dataset_path = os.path.join(data_path, f"train.{src_lang}-{tgt_lang}.{tgt_lang}")

    tgt_dataset = [dictionary.string(tokens) for tokens in MMapIndexedDataset(tgt_dataset_path)]
    src_dataset = [dictionary.string(tokens) for tokens in MMapIndexedDataset(src_dataset_path)]

    datasets[f"{src_lang}-{tgt_lang}"] = (src_dataset, tgt_dataset)


overlap_indices = defaultdict(list)

for pair1, (_, pair1_tgt) in datasets.items():
    for pair2, (_, pair2_tgt) in datasets.items():
        if pair1 != pair2:
            
            print(f"Finding overlap for {pair1}-{pair2}...")

            pair2_tgt_sents = set(pair2_tgt)

            for idx1, pair1_tgt_sent in enumerate(pair1_tgt):
                    if pair1_tgt_sent in pair2_tgt_sents:
                        overlap_indices[f"{pair1}_sents_in_{pair2}"].append(idx1)


# Store the dictionary to disk
with open('overlap_indices.pkl', 'wb') as f:
    pickle.dump(dict(overlap_indices), f)
