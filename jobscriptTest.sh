#!/usr/bin/env bash

#SBATCH -A C3SE508-19-3
#SBATCH -p chair
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1      
#SBATCH --job-name=goesrain

source ${HOME}/geostat-masters-thesis/setup_vera.sh

#export PYTHONPATH=$PYTHONPATH:"${HOME}/geostat-masters-thesis/results/models"

#Copy data to node
rsync -rq "${HOME}/geostat-masters-thesis/dataset/data/dataset-boxes" $TMPDIR

TEST_DATA="$TMPDIR/dataset-boxes"

# Execute this script in the node
python -u ${HOME}/geostat-masters-thesis/src/test.py -s "${HOME}/geostat-masters-thesis/results/test/" -M "${HOME}/geostat-masters-thesis/results/models/xception64_[100]_0.01__boxes_100_0.01_0_t5412_v1354[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622288705.386947.pckl" "${HOME}/geostat-masters-thesis/results/models/singles_fc32786_[100]_0.001__singles_100_0.001_0_t83360758_v20805499[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622293711.867882.pckl" -st "${HOME}/geostat-masters-thesis/dataset/data/dataset-boxes/train/stats.npy" -p ${TEST_DATA}/"$@"

