#!/bin/bash

echo "Type password followed by [ENTER]"

read -sp 'Password: ' password

for file in ./linkfiles/*
do
    echo "$file"
    python gpm_downloads.py -lf $(basename $file)
    #expect "assword: "   # matches both 'Password' and 'password'
    #send "$password\r";
    #interact

    
done

