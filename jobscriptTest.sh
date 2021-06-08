#!/usr/bin/env bash

#SBATCH -A C3SE508-19-3
#SBATCH -p chair
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1      
#SBATCH --job-name=goesrain

source ${HOME}/geostat-masters-thesis/setup_vera.sh

export PYTHONPATH=$PYTHONPATH:"${HOME}/geostat-masters-thesis/results/models"

#Copy data to node
rsync -rq "${HOME}/geostat-masters-thesis/dataset/data/dataset-boxes" $TMPDIR

TEST_DATA="$TMPDIR/dataset-boxes/train"

# Execute this script in the node
python -u ${HOME}/geostat-masters-thesis/src/test.py -p ${TEST_DATA} -s "${HOME}/geostat-masters-thesis/results/test/" -M "${HOME}/geostat-masters-thesis/results/models/xception64_[100]_0.01__boxes_100_0.01_0_t5412_v1354[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622288705.386947.pckl"

