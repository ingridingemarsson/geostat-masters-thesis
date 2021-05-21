#!/usr/bin/bash

echo "Please enter your pansat user password:"
echo "Password: "

read -s password

export PANSAT_PASSWORD=$password

linkfilepath='./files/links/linkfiles/linkfile2018-02*'

for file in $linkfilepath
do
    echo "$file"
    python make_dataset.py -lf $file
    echo "$file done"
    cd ../../visualize
    python plot_dataset.py -l $(basename $file) 
    cd ../dataset/downloads
    echo "$file plotted"
done

