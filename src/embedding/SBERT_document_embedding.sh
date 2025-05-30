#!/bin/bash
#SBATCH --time=0:30:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=alpha
#SBATCH --mem-per-cpu=20000M
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -A p_ml_il
#SBATCH -J "first_script"
#SBATCH -o "first_script-%j.out"
#SBATCH --error="first_script-%j.err"
#SBATCH --mail-user==manuel.berger@mailbox.tu-dresden.de
#SBATCH --mail-type ALL

module load release/25.06
module load GCCcore/13.3.0
module load Python/3.12.3

source /data/horse/ws/mabe761d-test-workspace/bin/activate

python SBERT_document_embedding.py

deactivate