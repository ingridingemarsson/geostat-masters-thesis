
# Setup node
ml purge
ml load GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.7.1-Python-3.7.4 TensorFlow/2.3.1-Python-3.7.4 
ml load foss/2019b Python/3.7.4 SciPy-bundle/2019.10-Python-3.7.4 matplotlib/3.1.1-Python-3.7.4 h5py/2.10.0-Python-3.7.4

# Add my files to python path 
export PYTHONPATH=$PYTHONPATH:"${HOME}/geostat-masters-thesis/src":"${HOME}/geostat-masters-thesis/visualize"
