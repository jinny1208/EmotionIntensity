#!/bin/bash
#SBATCH --job-name=mead_base
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --output=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.out
#SBATCH --error=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 1-mila

cd /home/mila/j/jeony/EmotionIntensity


python train.py -c configs/vctk_base.json -m 1-Mead_base \
  -o /home/mila/j/jeony/scratch/EmotionIntensity_runs