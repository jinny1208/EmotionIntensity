#!/bin/bash
#SBATCH --job-name=mead_base
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=80:00:00
#SBATCH --output=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.out
#SBATCH --error=/home/mila/j/jeony/scratch/EmotionIntensity_runs/logs/%x-%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 1-mila

cd /home/mila/j/jeony/EmotionIntensity


python inference.py \
    --config    configs/vctk_base.json \
    --checkpoint /home/mila/j/jeony/scratch/EmotionIntensity_runs/1-Mead_base/G_100000.pth \
    --filelist  /home/mila/j/jeony/EmotionIntensity/filelists/test_MEAD_filelist_espeak_with_attrs.txt \
    --output_dir /home/mila/j/jeony/scratch/EmotionIntensity_runs/1-Mead_base/test_wav_output