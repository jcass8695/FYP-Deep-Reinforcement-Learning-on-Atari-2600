#!/bin/sh
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH -p compute
#SBATCH -J spaceinvaders_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacassid@tcd.ie

. /etc/profile.d/modules.sh
module load cports7 Python/3.6.4-gnu
module load cports7 gcc/6.4.0-gnu

srun ./spaceinvaders.py dqn 500000 5000
