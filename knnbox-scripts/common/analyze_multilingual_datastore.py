import argparse
import pickle
import numpy as np
from collections import Counter, defaultdict


def analyze_ml_datastore(path, beam_size=5):
	k = int(args.path.split("k_")[-1].split("-")[0])

	with open(f"{path}/stats.pickle", "rb") as f:
		stats = pickle.load(f)

	# distances = []
	# for i in range(len(stats["distances"])):
	# 	if i%100 == 0:
	# 		print(i, stats["distances"][i])
	# 	distances.append(round(stats["distances"][i]))

	print(stats.keys())

	assert len(stats["nn_top5_tok"]) == len(stats["combi_top5_tok"])

	print(len(stats["combi_top5_tok"]))

	identical, identical_sorted = [0]*len(stats["nn_top5_tok"]), [0]*len(stats["nn_top5_tok"])
	changes_per_src_lang = defaultdict(int)
	for idx, (nn_toks, combi_toks) in enumerate(zip(stats["nn_top5_tok"], stats["combi_top5_tok"])):

		if nn_toks == combi_toks:
			identical[idx] = 1
		
		if sorted(nn_toks) == sorted(combi_toks):
			identical_sorted[idx] = 1

		# actual changes in prediction
		else:
			changes_per_src_lang[stats["src_lang_tags"][idx]] += 1

		# if "</s>" in nn_toks:
		# if True:
		# 	print(nn_toks==combi_toks, set(nn_toks)==set(combi_toks),nn_toks, combi_toks, stats["nn_top5_prob"][idx], stats["combi_top5_prob"][idx])

	print("identical: ", sum(identical), "identical (after sorting): ", sum(identical_sorted), "total: ", len(stats["nn_top5_tok"]))


	print("percentage changed:                 ",(1-sum(identical)/len(stats["nn_top5_tok"])))
	print("percentage changed (after sorting): ",(1-sum(identical_sorted)/len(stats["nn_top5_tok"])))

	for lang in set(stats["src_lang_tags"]):
		print(lang, changes_per_src_lang[lang]/sum(changes_per_src_lang.values()))


	quit()

	# tokens now look like this: dict_keys(['distances', 'vals', 'sentence_ids', 'src_lang_tags', 'tgt_lang_tags', 'token_positions', 'ds_top5_tok', 'ds_top5_prob', 'nn_top5_tok', 'nn_top5_prob'])
	# assert all([len(stats[key]) == len(stats["knn_prob_top_1"]) for key in ["knn_prob_top_1", "neural_prob_top_1", "neural_prob_top_5"]])
	# assert len(stats["knn_prob_top_1"]) > 0

	# agreed_top_1, knn_in_neural_5 = 0, 0
	# for knn_prob, neural_prob, neural_prob_5 in zip(stats["knn_prob_top_1"], stats["neural_prob_top_1"], stats["neural_prob_top_5"]):
	# 	if knn_prob == neural_prob:
	# 		agreed_top_1 += 1
	# 	if knn_prob in neural_prob_5:
	# 		knn_in_neural_5 += 1

	# agreed_top_1 = agreed_top_1/len(stats["knn_prob_top_1"])
	# knn_in_neural_5 = knn_in_neural_5/len(stats["knn_prob_top_1"])
	# print(f"Same top prediction knn / neural: {round(agreed_top_1, 2)}")
	# print(f"knn prediction in top 5 neural: {round(knn_in_neural_5, 2)}")
	# print()

	assert all([len(stats[key]) == len(stats["vals"]) for key in ["distances", "vals", "sentence_ids", "src_lang_tags", "tgt_lang_tags", "token_positions"]])
	assert len(stats["vals"]) % k == 0

	# How many unique vals are there per knn-call?
	print("Number of unique `values` per kNN-call:")
	count_unique_vals = []
	for i in range(0, len(stats["vals"]), k):
		count_unique_vals.append(len(set(stats["vals"][i:i+k])))


	sorted_count_unique_vals = sorted(Counter(count_unique_vals).items(), key=lambda item: item[1], reverse=True)

	for key, value in sorted_count_unique_vals:
		print(f"{key}/{k} unique: {round(value/len(count_unique_vals), 2)}%")
	print()

	# Which source language tags are used?
	print("Distribution of source language use:")
	count_src_lang_tags = Counter(stats["src_lang_tags"])
	for key in count_src_lang_tags.most_common(len(count_src_lang_tags)):
		print(key[0])
	for key in count_src_lang_tags.most_common(len(count_src_lang_tags)):
		print(key[1]/sum(count_src_lang_tags.values()))
	print()

	# Which source language tags are used?
	print("Distribution of target language use:")
	count_tgt_lang_tags = Counter(stats["tgt_lang_tags"])
	for key in count_tgt_lang_tags.most_common(len(count_tgt_lang_tags)):
		print(key[0], key[1]/sum(count_tgt_lang_tags.values()))
	print()

	# Most common token positions used in knn
	print("Most common `vals` (target token) locations:")
	count_tok_pos = Counter(stats["token_positions"])
	for key in count_tok_pos.most_common(30):
		print(key[0], key[1]/sum(count_tok_pos.values()))
	print()

	# Most common words predicted by knn
	print("Most common target token predictions:")
	count_vals = Counter(stats["vals"])
	for key in count_vals.most_common(50):
		print(key[0], key[1]/sum(count_vals.values()))
	print()

	print("Average distance:")
	# print(f"{np.array(stats["distances"]).mean()} (std {np.array(stats["distances"]).std()})")
	print(np.array(stats["distances"]).mean(), "std=", np.array(stats["distances"]).std())


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, help="Path to multilingual datastore.")

	parser.add_argument
	args = parser.parse_args()

	analyze_ml_datastore(path=args.path)