#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=51200MB
#SBATCH --time=06:20:00
#SBATCH -p economy
#SBATCH --output=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleLog.txt
#SBATCH --error=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleEroor.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr4xb@virginia.edu

cd /nv/blue/mr4xb/public_html/Nahid_IR_Base/
module load java/1.8.0_45
java -jar ./LRHTSM_camera_newegg_aspect.jar
java -jar ./LRHTSM_camera_newegg_aspectsenti.jar
java -jar ./LRHTSM_camera_nonewegg_aspectsenti.jar
java -jar ./LRHTSM_tablet_newegg_aspect.jar
java -jar ./LRHTSM_tablet_newegg_aspectsenti.jar
java -jar ./LRHTSM_tablet_nonewegg_aspectsenti.jar
java -jar ./LRHTSM_phone_newegg_aspect.jar
java -jar ./LRHTSM_phone_newegg_aspectsenti.jar
java -jar ./LRHTSM_phone_nonewegg_aspectsenti.jar
java -jar ./LRHTSM_tv_newegg_aspect.jar
java -jar ./LRHTSM_tv_newegg_aspectsenti.jar
java -jar ./LRHTSM_tv_nonewegg_aspectsenti.jar
module unload java/1.8.0_45
