#!/usr/bin/bash

echo "Please enter your pansat user password:"
echo "Password: "

read -s password

export PANSAT_PASSWORD=$password

for file in ./downloads/links/linkfiles/*
do
    echo "$file"
    python make_dataset.py -lf $(basename $file) --rem True -s ~/Dendrite/UserAreas/Ingrid/Dataset
    echo "$file done"
done


