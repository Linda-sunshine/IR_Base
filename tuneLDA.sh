#!/bin/bash
read -p "Enter source, crossV, model:" source crossV model
echo
for k in {5..50..5}
do
    nohup ./run -prefix ~/lab/dataset -source $source -set byUser_20k_review -topicmodel $model -crossV $crossV -nuOfTopics $k  -emIter 50 > ./output/"$crossV"_"$source"_"$model"_"$k".output 2>&1 &
done