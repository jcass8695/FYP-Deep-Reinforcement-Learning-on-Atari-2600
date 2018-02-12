#!bin/sh
#SBATCH -n 16           # 16 cores
#SBATCH -t 0-13:00:00   # 13 hours
#SBATCH -p gu-shared      # partition name
#SBATCH -J spaceinvaders_train  # sensible name for the job

. /etc/profile.d/modules.sh
module load cports7 Python/3.6.4-gnu
module load cports7 gcc/6.4.0-gnu

mpirun ./spaceinvaders.py
