#!/bin/sh
#SBATCH --time=20:00:00
#SBATCH --mem=48G
#SBATCH --job-name=nickelforward
#SBATCH --partition=cse896
#SBATCH --gres=gpu
#SBATCH --error=job.qa.%J.err
#SBATCH --output=job.qa.%J.out
#SBATCH --output=job.qa.%J.out

module load anaconda
conda activate project

python -u $@

