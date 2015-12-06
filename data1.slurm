#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=51200MB
#SBATCH --time=06:20:00
#SBATCH -p economy
#SBATCH --output=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleLogASUM.txt
#SBATCH --error=/nv/blue/mr4xb/public_html/Nahid_IR_Base/output/consoleEroorASUM.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mr4xb@virginia.edu

cd /nv/blue/mr4xb/public_html/Nahid_IR_Base/
module load java/1.8.0_45
java -jar ./tablet_3000.jar
java -jar ./tv_3000.jar
module unload java/1.8.0_45
