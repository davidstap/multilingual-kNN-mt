from knnbox.datastore.utils import build_faiss_index
from collections import defaultdict
import argparse
import json
import os
import numpy as np


def combine_datastores(path: str, pairs: list, map: str, save_path: str = None):
	pairs = [pair.replace("-", "_") for pair in pairs]

	keys, vals, configs = [], [], []

	if map != "":
		map = "_" + map

	auxiliary_info = defaultdict(list)
	for pair in pairs:
		ds_path = f"{path}/data-knnds-{pair}{map}"
		with open(f"{ds_path}/config.json") as f:
			config = json.loads(f.read())
			configs.append(config)
			ds_size, keys_dim = config["data_infos"]["keys"]["shape"]
			keys_dtype = np.float16 if "float16" in config["data_infos"]["keys"]["dtype"] else np.float32
			print(f"ds size {pair}:    {ds_size}")

		keys.append(np.memmap(f"{ds_path}/keys.npy", dtype=keys_dtype, mode='r', shape=(ds_size, keys_dim)))
		vals.append(np.memmap(f"{ds_path}/vals.npy", dtype=int,        mode='r', shape=(ds_size, 1)))


		for f in ["sentence_ids", "src_lang_tags", "tgt_lang_tags", "token_positions"]:
			fn = f"{ds_path}/{f}.npy"
			if os.path.isfile(fn):
				auxiliary_info[f].append(np.memmap(f"{ds_path}/{f}.npy", dtype=int, mode='r', shape=(ds_size, 1)))

			else:
				print("File {fn} does not exist; cannot combine.")

	ds_sizes = [cfg["data_infos"]["vals"]["shape"][0] for cfg in configs]

	print(f"combined ds size: {sum(ds_sizes)}")

	if save_path == None:
		save_path = path+"/data-knnds-"+"+".join(pairs)
	os.makedirs(save_path, exist_ok=True)

	combined_config = configs[0]
	combined_config["data_infos"]["vals"]["shape"] = [sum(ds_sizes)]
	combined_config["data_infos"]["keys"]["shape"] = [sum(ds_sizes), keys_dim]

	for k in auxiliary_info.keys():
		combined_config["data_infos"][k]["shape"] = [sum(ds_sizes)]


	with open(f"{save_path}/config.json", "w") as f:
		cfg_str = json.dumps(combined_config, indent=4)
		print(cfg_str, file=f)

	# create new memmap objects with the correct shape
	combined = {}
	combined["vals"] = np.memmap(f"{save_path}/vals.npy", dtype=int, mode='w+', shape=(sum(ds_sizes), 1))
	combined["keys"] = np.memmap(f"{save_path}/keys.npy", dtype=keys_dtype, mode="w+", shape=(sum(ds_sizes), keys_dim))
	for k in auxiliary_info.keys():
		combined[k] = np.memmap(f"{save_path}/{k}.npy", dtype=int, mode='w+', shape=(sum(ds_sizes), 1))

	# iterate over each existing memmap object, copying its data into the new object
	def write_to_memmap(memmap_list, combined_memmap, name, offset=0, chunk_size=10000):
		for x in memmap_list:
			# iterate over each chunk of data in the existing memmap object
			for i in range(0, x.shape[0], chunk_size):
				start, cease = i, min(i + chunk_size, x.shape[0])
				print(f"Processing {name}: {offset+start}/{sum(ds_sizes)} ({(offset+start)/sum(ds_sizes)}%)")
				# copy the chunk of data into the new memmap object
				combined_memmap[offset+start:offset+cease] = x[start:cease]
			# update the offset for the next memmap object
			offset += x.shape[0]


	write_to_memmap(memmap_list=vals, combined_memmap=combined["vals"], name="vals")
	write_to_memmap(memmap_list=keys, combined_memmap=combined["keys"], name="keys")

	for k in auxiliary_info.keys():
		print("do ",k)
		write_to_memmap(memmap_list=auxiliary_info[k], combined_memmap=combined[k], name=k)

	# Build FAISS index
	keys = np.memmap(
		f"{save_path}/keys.npy",
		dtype=np.float16,
		mode="r",
		shape=(sum(ds_sizes), keys_dim)
	)
	build_faiss_index(
		keys=keys,
		shape=(sum(ds_sizes), keys_dim),
		output_filename=f"{save_path}/keys.faiss_index"
	)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, help="Path to datastores.")
	parser.add_argument('--pairs', nargs='+', type=str, help='Language pairs that should be combined.')
	parser.add_argument("--xlingual_mapping", type=str, default="")
	parser.add_argument("--save_path", default=None, type=str, help="Path of combined datastore")

	parser.add_argument
	args = parser.parse_args()

	assert args.path and args.pairs
	assert len(args.pairs) > 1

	combine_datastores(path=args.path, pairs=args.pairs, map=args.xlingual_mapping, save_path=args.save_path)


# srun --mem=64G --cpus-per-task=16 --time=48:00:00 python knnbox-scripts/common/combine_datastores.py --path $DATA/fairseq_ted_m2m_100


# srun --mem=32G --partition=gpu --gres=gpu:pascalxp:1 --cpus-per-task=16 --time=48:00:00 python knnbox-scripts/common/combine_datastores.py \
# 	--path $DATA/fairseq_ted_m2m_100 \
#	--pairs ka_en hy_en sq_en el_en \
#	--xlingual_mapping 2ka \
#	--save_path $DATA/fairseq_ted_m2m_100/data-knnds-greek_en_2ka \