#!/usr/bin/env bash

#SBATCH -A C3SE508-19-3
#SBATCH -p chair
#SBATCH -t 0-00:10:00 # How long?
#SBATCH --gres=gpu:1      

# Setup node
ml purge
ml load GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.7.1-Python-3.7.4
ml load foss/2019b Python/3.7.4 SciPy-bundle/2019.10-Python-3.7.4 matplotlib/3.1.1-Python-3.7.4 h5py/2.10.0-Python-3.7.4

# Install quantnn
python -m pip install --user -e $HOME/Documents/Git/quantnn

# Install torchvision that does not work with system module loader
python -m pip install --user torchvision==0.4.0

# Add my files to python path 
# Here should be included: load_data.py, model.py and train.py
export PYTHONPATH=$PYTHONPATH:"$HOME/geostat-masters-thesis/src"

# Data #Copy data to node
#cp -r "$HOME/geostat-masters-thesis/dataset/data/dataset-singles" $TMPDIR
DATA = "$HOME/geostat-masters-thesis/dataset/data/dataset-singles" 


# Execute the arguments to this script in the node
#"$@"
#python test_run.py -p $TMPDIR/dataset-singles -s $TMPDIR/results
python test_run.py -p $DATA -s "$HOME/geostat-masters-thesis/results"
