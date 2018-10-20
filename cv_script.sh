#!/bin/sh
cur_time=$(date +"%s")
data=YelpNew
em_iter=50
nu_topics=5
var_iter=10
dim=10
save_prefix=./data/embeddingExp/eub/$cur_time\_$data\_emIter\_$em_iter\_nuTopics_$nu_topics\_varIter_$var_iter\_dim_$dim
echo $save_prefix
mkdir -p $save_prefix
for k in 0 1 2 3 4
	do nohup ./run -prefix ./data/CoLinAdapt -emIter $em_iter -nuTopics $nu_topics -varIter $var_iter -dim $dim -savePrefix $save_prefix > ./data/embeddingExp/1020Eub_$data\_emIter_$em_iter\_nuTopics_$nu_topics\_varIter_$var_iter\_dim_$dim\_fold_$k.output &
done
