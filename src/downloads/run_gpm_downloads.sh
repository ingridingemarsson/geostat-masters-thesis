#!/usr/bin/bash

echo "Please enter your pansat user password:"
echo "Password: "

read -s password

export PANSAT_PASSWORD=$password

for file in ./links/linkfiles/*
do
    echo "$file"
    python make_dataset.py -lf $(basename $file) -t True --rem True
    echo "$file done"
done


