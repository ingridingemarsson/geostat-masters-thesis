#!/usr/bin/env bash

#SBATCH -A C3SE508-19-3
#SBATCH -p chair
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1      
#SBATCH --job-name=goesrain

source ${HOME}/geostat-masters-thesis/setup_vera.sh

#Copy data to node
rsync -rq "${HOME}/geostat-masters-thesis/dataset/data/dataset-boxes" $TMPDIR
myimage="${HOME}/geostat-masters-thesis/dataset/data/image_to_plot/" 

TRAINING_DATA="$TMPDIR/dataset-boxes/train"
VALIDATION_DATA="$TMPDIR/dataset-boxes/validation"

# Execute this script in the node
#python -u ${HOME}/geostat-masters-thesis/src/train.py -p "$TMPDIR/dataset-boxes/" -s "${HOME}/geostat-masters-thesis/results/" -D "boxes" "$@" 
python -u ${HOME}/geostat-masters-thesis/src/train.py -p ${TRAINING_DATA} ${VALIDATION_DATA} -s "${HOME}/geostat-masters-thesis/results/" --im_to_plot ${myimage} -D "boxes" "$@" 

