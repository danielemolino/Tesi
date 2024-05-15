#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-493 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4  # qui selezioni il tipo di gpu ed il numero
#SBATCH -t 0-00:10:00  # tempo di calcolo (massimo 167 ore == 7 giorni)
# Output files
#SBATCH --error=./CoDI-Original/error/job_%J.err  # file di error
#SBATCH --output=./CoDI-Original/output/out_%J.out  # file di output
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniele.molino@alcampus.it  # qui la tua mail

# Load modules
module load virtualenv/20.23.1-GCCcore-12.3.0 matplotlib/3.7.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a h5py/3.9.0-foss-2023a


# Activate venv
cd /mimer/NOBACKUP/groups/naiss2023-6-336/dmolino/venv5
source bin/activate

# Executes the code (path alla tua directory)
cd /mimer/NOBACKUP/groups/naiss2023-6-336/dmolino/CoDI-Original

# Train HERE YOU RUN YOUR PROGRAM
python3 demo.py
deactivate
