#!/bin/bash -l
#SBATCH --job-name=mpi_denoising       # Job name
#SBATCH --output=logs/%x-%j.out       # Standard output log (%x=job name, %j=job ID)
#SBATCH --error=logs/%x-%j.err        # Error log
#SBATCH --time=01:00:00               # Time limit (HH:MM:SS)
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --ntasks-per-node=4           # MPI tasks per node
#SBATCH --cpus-per-task=1             # CPUs per task

# Load modules and activate Python environment
module purge
module load lang/Python               # Load Python module
module load mpi/OpenMPI               # Load OpenMPI module
source .venv/bin/activate             # Activate your virtual environment

# Create output directories if they don't exist
mkdir -p logs plots

# Run your MPI program
mpiexec -n 8 python3 parallel_denoising.py