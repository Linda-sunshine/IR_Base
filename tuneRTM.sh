#!/bin/bash
read -p "Enter source, model, mode:" source model mode
echo
nohup ./runRTM -prefix ~/lab/dataset -source $source -set byUser_20k_review -topicmodel $model -mode $mode > ./output_data/"$mode"_"$source"_"$model".output 2>&1 &