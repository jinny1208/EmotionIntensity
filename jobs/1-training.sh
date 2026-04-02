#!/bin/bash
#SBATCH --job-name=vctk_base
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=100:00:00
#SBATCH --output=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.out
#SBATCH --error=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 1-mila

cd /home/mila/j/jeony/EmotionIntensity

## For resuming
# python train.py -c configs/vctk_base.json -m vctk_base \
#   -o /home/mila/j/jeony/scratch/EmotionIntensity_runs

python train.py -c configs/vctk_base.json -m vctk_base 