#!/bin/bash

echo "Type password followed by [ENTER]"

read -sp 'Password: ' password

for file in ./linkfiles/*
do
    echo "$file"
    python gpm_downloads.py -lf $(basename $file)


    
done

