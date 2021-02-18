#!/usr/bin/bash

python get_word.py

for file in ./links/linkfiles/*
do
    echo "$file"
    python make_dataset.py -lf $(basename $file) -t True -b True -rem True
    echo "$file done"
done

python remove_word.py

