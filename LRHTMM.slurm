#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=51200MB
#SBATCH --time=04:20:00
#SBATCH -p economy
#SBATCH --output=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleLog.txt
#SBATCH --error=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleEroor.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr4xb@virginia.edu

cd /nv/blue/mr4xb/public_html/Nahid_IR_Base/
module load java/1.8.0_45
java -jar ./LRHTMM_camera_0.jar
java -jar ./LRHTMM_camera_1000.jar
java -jar ./LRHTMM_camera_2000.jar
java -jar ./LRHTMM_camera_3000.jar
java -jar ./LRHTMM_camera_4000.jar
java -jar ./LRHTMM_camera_5000.jar
java -jar ./LRHTMM_tablet_0.jar
java -jar ./LRHTMM_tablet_1000.jar
java -jar ./LRHTMM_tablet_2000.jar
java -jar ./LRHTMM_tablet_3000.jar
java -jar ./LRHTMM_tablet_4000.jar
java -jar ./LRHTMM_tablet_5000.jar
java -jar ./LRHTMM_phone_0.jar
java -jar ./LRHTMM_phone_1000.jar
java -jar ./LRHTMM_phone_2000.jar
java -jar ./LRHTMM_phone_3000.jar
java -jar ./LRHTMM_phone_4000.jar
java -jar ./LRHTMM_phone_5000.jar
java -jar ./LRHTMM_tv_0.jar
java -jar ./LRHTMM_tv_1000.jar
java -jar ./LRHTMM_tv_2000.jar
java -jar ./LRHTMM_tv_3000.jar
java -jar ./LRHTMM_tv_4000.jar
java -jar ./LRHTMM_tv_5000.jar
module unload java/1.8.0_45