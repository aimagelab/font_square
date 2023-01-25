#!/bin/bash
#SBATCH --job-name=font_square_gen
#SBATCH --output=/homes/fquattrini/fontsquare/job_logs/generation_%j.out
#SBATCH --error=/homes/fquattrini/fontsquare/job_logs/generation_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=prod
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:0
#SBATCH --array=0-103
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=4096
#SBATCH --begin=now+2hour

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /homes/fquattrini/.conda/envs/fontsquare
cd /homes/fquattrini/fontsquare

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " "$SLURM_ARRAY_JOB_ID"

sleep $((RANDOM % 60))

srun python save_dataset.py --out_dir /mnt/beegfs/work/FoMo_AIISDH/datasets/font_square --workers 2 --font_split_size=100 --font_split_id=$SLURM_ARRAY_TASK_ID
