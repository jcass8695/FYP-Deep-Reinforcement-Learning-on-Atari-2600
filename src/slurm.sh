#!/bin/sh
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH -p compute
#SBATCH -J si_dqn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jacassid@tcd.ie
#SBATCH --gres=gpu:2

. /etc/profile.d/modules.sh
module load cports6 Python/3.5.2-gnu
module load cports7 gcc/6.4.0-gnu
module load apps cuda/9.0
srun ./main.py space_invaders dqn
