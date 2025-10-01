#!/bin/bash
#SBATCH --job-name=gan_test
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=3
#SBATCH --output=logs/%x_%j.out


# source ~/miniconda3/etc/profile.d/conda.sh
source /gpfs/home/shaana01/miniconda3/etc/profile.d/conda.sh
conda activate cenv3

python c2n_mnist.py
# python sample.py
