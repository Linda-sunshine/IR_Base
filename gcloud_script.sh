#!/bin/sh
data=YelpNew
em_iter=50
nu_topics=40
var_iter=10
inner_iter=1
dim=10
ada=false
mode=cv4edge
save_prefix=/home/lin/embeddingExp/eub/$data\_emIter\_$em_iter\_nuTopics_$nu_topics\_varIter_$var_iter\_innerIter_$inner_iter\_dim_$dim\_ada_$ada\_mode_$mode
echo $save_prefix
mkdir -p $save_prefix
for k in 0 1 2 3 4  
	do 
		cur_time=$(date +"%s")
		save_dir=$save_prefix/fold_$k\_$cur_time
		echo $save_dir
 		nohup ./run -prefix /home/lin/DataSigir/ -data $data -mode $mode -emIter $em_iter -nuTopics $nu_topics -varIter $var_iter -innerIter $inner_iter -dim $dim -ada $ada -kFold $k -saveDir $save_dir > /home/lin/embeddingExp/1023Eub_$data\_emIter_$em_iter\_nuTopics_$nu_topics\_varIter_$var_iter\_innerIter_$inner_iter\_dim_$dim\_ada_$ada\_mode_$mode\_fold_$k.output 
done
