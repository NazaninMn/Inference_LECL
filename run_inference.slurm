#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=100000
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A gutintelligencelab
#SBATCH -o Llineage_tracing.out

module purge
module load singularity pytorch/1.8.1 

singularity run --nv $CONTAINERDIR/pytorch-1.8.1.sif main.py --nuclei_channel_image 3 --marker_channel_image 1 --root /home/nm4wu/ --test_list 90985-19B.tif --marker_name EdU



