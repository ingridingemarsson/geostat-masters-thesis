#!/usr/bin/env bash

#SBATCH -A C3SE508-19-3
#SBATCH -p chair
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1      
#SBATCH --job-name=goesrain

source ${HOME}/geostat-masters-thesis/setup_vera.sh

#Copy data to node
rsync -rq "${HOME}/geostat-masters-thesis/dataset/data/dataset-boxes" $TMPDIR

TEST_DATA="$TMPDIR/dataset-boxes/train"

# Execute this script in the node
python -u ${HOME}/geostat-masters-thesis/src/test.py -p ${TEST_DATA} -s "${HOME}/geostat-masters-thesis/results/test/" "$@" 

