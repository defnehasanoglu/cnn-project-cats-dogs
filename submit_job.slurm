#!/bin/bash
#SBATCH --job-name=catsdogs_cnn
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=ampere              
#SBATCH --gpus=1                       
#SBATCH --cpus-per-task=8               
#SBATCH --mem=32G                       
#SBATCH --time=01:00:00                
#SBATCH --mail-type=START,END,FAIL           
#SBATCH --mail-user=nchasan@bio.auth.gr  

module purge
module load gcc/14.2.0 python/3.11 cuda/12.6.2 cudnn miniconda3

source $CONDA_PROFILE/conda.sh

conda activate /mnt/appbio_a/groups/Year_4/NChasan/conda_envs/thesis

cd $SLURM_SUBMIT_DIR

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python3 train.py

echo "Job finished: $(date)"