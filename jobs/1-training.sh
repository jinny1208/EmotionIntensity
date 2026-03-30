#!/bin/bash
#SBATCH --job-name=vctk_base
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.out
#SBATCH --error=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 1-mila

cd /home/mila/j/jeony/EmotionIntensity


python train.py -c configs/vctk_base.json -m vctk_base \
  -o /home/mila/j/jeony/scratch/EmotionIntensity_runs