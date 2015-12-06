#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=51200MB
#SBATCH --time=10:20:00
#SBATCH -p economy
#SBATCH --output=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleLog.txt
#SBATCH --error=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleEroor.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr4xb@virginia.edu

cd /nv/blue/mr4xb/public_html/Nahid_IR_Base/
module load java/1.8.0_45
java -jar ./LRHTSM_tv_0.jar
java -jar ./LRHTSM_tv_3000.jar
java -jar ./LRHTSM_phone_0.jar
java -jar ./LRHTMM_tv_0.jar
java -jar ./LRHTMM_phone_0.jar
java -jar ./LDA_tv_3000.jar
module unload java/1.8.0_45
