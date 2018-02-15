#!/bin/sh
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH -p compute
#SBATCH -J $1_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacassid@tcd.ie
#SBATCH --gres=gpu:2

. /etc/profile.d/modules.sh
module load cports7 Python/3.6.4-gnu
module load cports7 gcc/6.4.0-gnu
module load apps cuda/8.0
srun ./main.py $1 $2 500000 1000 $3
