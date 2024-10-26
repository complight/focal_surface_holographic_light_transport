#!/bin/bash
#SBATCH -J Job
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64Gb




python test.py 
