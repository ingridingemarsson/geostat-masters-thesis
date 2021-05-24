#!/bin/bash

echo "Please enter your pansat user password:"
echo "Password: "

read -s password

export PANSAT_PASSWORD=$password

linkfilepath='./files/links/linkfiles/linkfile20*'
datalocpath='/home/ingrid/geostat-masters-thesis/dataset/origin'
datatemppath='/home/ingrid/geostat-masters-thesis/dataset/temp'
datasavepath='/home/ingrid/Dendrite/UserAreas/Ingrid/Dataset'

for file in $linkfilepath
do
    echo "$file"
    python make_dataset.py -lf $file --temp "$datatemppath"/"$( basename "$file" .txt )"/ 
    echo "$file done"
    cp -r "$datalocpath"/"$( basename "$file" .txt )"/ "$datasavepath"/
    rm -r "$datalocpath"/"$( basename "$file" .txt )"/
    echo "$file moved"
done


