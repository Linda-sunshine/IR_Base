#!/bin/bash
read -p "Enter source, model, mode, coldstart: " source model mode cold
echo
nohup ./runRTM -prefix ~/lab/dataset -source $source -set byUser_20k_review -topicmodel $model -mode $mode -flagColdstart $cold > ./output_data/"$mode"_"$cold"_"$source"_"$model".output 2>&1 &