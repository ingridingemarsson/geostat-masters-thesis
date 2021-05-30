#!/bin/bash

datatemppath='/home/ingrid/geostat-masters-thesis/reference/temp_gauge'
datalocpath='/home/ingrid/geostat-masters-thesis/reference/rain_gauge_preds'
datasavepath='/home/ingrid/Dendrite/UserAreas/Ingrid/rain_gauge_preds'

for day in {1..31}
do
    echo $day
    python make_rain_gauge_preds.py --day $day --temp_path "$datatemppath/$day/" --save_path "$datalocpath/$day/"
    echo "$day done"
    cp -r "$datalocpath/$day/" "$datasavepath/$day/"
    rm -r "$datalocpath/$day/"
    echo "$day moved"
done


