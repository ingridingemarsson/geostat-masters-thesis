#!/usr/bin/bash

echo "Please enter your pansat user password:"
echo "Password: "

read -s password

export PANSAT_PASSWORD=$password

linkfilepath='./files/links/linkfiles/linkfile2018-02*'
datalocpath='~/Git/geostat-masters-thesis/src/dataset/origin/'
datasavepath='~/Dendrite/UserAreas/Ingrid/origin'

for file in $linkfilepath
do
    echo "$file"
    python make_dataset.py -lf $file
    echo "$file done"
    mv "$datalocpath"/"$( basename "$linkfilepath" .txt )"/*  "$datasavepath"/"$( basename "$linkfilepath" .txt )"/
    echo "$file moved"
done

