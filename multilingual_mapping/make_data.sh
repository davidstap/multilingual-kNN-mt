#!/bin/bash

#SBATCH --job-name=W
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --export=SRC1="be",SRC2="ru"

echo "Training mapping for '$SRC1' into '$SRC2'."

CODE_PATH=/home/dstap1/code/knn-multilingual

source activate knn

python -u $CODE_PATH/multilingual_mapping/make_data.py --src1 ${SRC1} --src2 ${SRC2}