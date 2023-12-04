# Multilingual _k_-Nearest-Neighbor Machine Translation
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

This repo contains code for the paper [Multilingual _k_-Nearest-Neighbor Machine Translation](https://arxiv.org/abs/2310.14644) (EMNLP 2023).

## Installation
Our code builds on the [knn-box toolkit](https://github.com/NJUNLP/knn-box/tree/master). You can install the toolkit and our extensions by:

```shell
conda create -n multilingual-knn python=3.8
conda activate multilingual-knn

git clone git@github.com:davidstap/multilingual-kNN-mt.git
cd multilingual-kNN-mt
pip install --editable ./
# Installing faiss with pip is not recommended, so we use conda
conda install faiss-gpu -c pytorch
```

## Datastore creation
build bilingual datastore
create multilingual datastor

### 1. Download pretrained models and dataset
You can prepare pretrained models and dataset by executing the following command:

```shell
cd knnbox-scripts
bash prepare_dataset_and_model.sh
```

It is straightforward to make changes to this script to change model and/or datasets.

### 2. Build *k*NN-MT Datastore

After preparation, the next step is to build a datastore. The following script shows how to build a datastore from Hebrew (he) into English (en), using a multilingual model that only supports Hebrew and Arabic in and out of English directions. The result is a datastore, which is stored in the `data-knnds-he_en` folder. To create datastores for other languages, simply change the variables.

```shell
SRC=he
TGT=en
DATA_PATH=/path/to/preprocessed/dataset
MODEL_PATH=/path/to/preprocessed/model

CUDA_VISIBLE_DEVICES=0 python knnbox-scripts/common/validate.py $DATA_PATH \
    --task translation_multi_simple_epoch \
    --langs en,he \
    --lang-pairs en-ar,en-he,ar-en,he-en \
    --seed 222 \
    --source-lang $SRC \
    --target-lang $TGT \
    --path $MODEL_PATH/best.pt \
    --model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
    --dataset-impl mmap \
    --valid-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --user-dir knnbox/models \
    --arch vanilla_knn_mt@transformer \
    --knn-mode build_datastore \
    --knn-datastore-path $DATA_PATH/data-knnds-${SRC}_${TGT} \
    --share-decoder-input-output-embed \
    --decoder-langtok \
    --encoder-langtok src \
    --fp16 \
    --dropout 0.1 \
    --label-smoothing 0.1
```

Bilingual datastores can be combined, resulting in a multilingual datastores. This leads to better performance. The following script can be used to combine he-en and ar-en:

```shell
DATA_PATH=/path/to/preprocessed/dataset

python knnbox-scripts/common/combine_datastores.py \
    --path $DATA_PATH \
    --pairs ar_en he_en \
    --save_path $DATA_PATH/data-knnds-ar_en-he_en \
```

### 3. Do *k*NN-MT inference.
Use the following script to do inference using *k*NN, and calculate BLEU scores:

```shell
SRC=he
TGT=en
DATA_PATH=/path/to/preprocessed/dataset
MODEL_PATH=/path/to/preprocessed/model
RESULTS_PATH=${MODEL_PATH}/generations/gen-test-${SRC}_${TGT}-${DS}-k_${KNN_K}-l_${KNN_LAMBDA}-t_${KNN_TEMP}

KNN_K=16
KNN_LAMBDA=0.5
KNN_TEMP=100

CUDA_VISIBLE_DEVICES=0 python knnbox-scripts/common/generate.py $DATA_PATH \
    --task translation_multi_simple_epoch \
    --langs en,he \
    --lang-pairs en-ar,en-he,ar-en,he-en \
    --source-lang $SRC \
    --target-lang $TGT \
    --remove-bpe 'sentencepiece' \
    --path $MODEL_PATH/best.pt \
    --share-decoder-input-output-embed \
    --results-path $RESULTS_PATH \
    --dataset-impl mmap \
    --beam 5 \
    --gen-subset test \
    --max-tokens 1024 \
    --user-dir knnbox/models \
    --arch vanilla_knn_mt@transformer \
    --knn-mode inference \
    --knn-datastore-path $DATA_PATH/data-knnds-${SRC}_${TGT} \
    --knn-k $KNN_K \
    --knn-lambda $KNN_LAMBDA \
    --knn-temperature $KNN_TEMP \
    --model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
    --decoder-langtok \
    --encoder-langtok src \
    --skip-invalid-size-inputs-valid-test \
    --sampling-method temperature \
    --label-smoothing 0.1 \
    --criterion label_smoothed_cross_entropy \
    --fp16 \
    --seed 222

grep ^T $RESULTS_PATH/generate-test.txt | LC_ALL=C sort -V | cut -f2- > $RESULTS_PATH/ref.txt
grep ^D $RESULTS_PATH/generate-test.txt | LC_ALL=C sort -V | cut -f3- > $RESULTS_PATH/hyp.txt
grep ^S $RESULTS_PATH/generate-test.txt | LC_ALL=C sort -V | cut -f2- > $RESULTS_PATH/src.txt

echo knn inference result for $SRC-$TGT K=$KNN_K L=$KNN_LAMBDA T=$KNN_TEMP saved at $RESULTS_PATH
sacrebleu $RESULTS_PATH/ref.txt -i $RESULTS_PATH/hyp.txt -m bleu | grep '"score"' | grep -oE '[0-9]+(\.[0-9]+)?' > $RESULTS_PATH/bleu.txt
cat $RESULTS_PATH/bleu.txt
```

* `KNN_K` is the number of neighbors, good values for the TED dataset are 8, 16, 32, 64
* `KNN_LAMBDA` is the weight of the *k*NN distribution, good values are \{0.2, 0.3, ..., 0.7\}
* `KNN_TEMP` is the temperature for the *k*NN distribution to make it more smooth. Good values are 10, 100.

Instead of fixing these values, one can do a hyperparameter search on the validation set, and use the best parameters to do decoding on the test set.

## Citation

If you use our code, please cite the following:

```bibtex
@misc{stap2023multilingualknn,
      title={Multilingual k-Nearest-Neighbor Machine Translation}, 
      author={David Stap and Christof Monz},
      booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
      year={2023},
      publisher = "Association for Computational Linguistics",
}
```
