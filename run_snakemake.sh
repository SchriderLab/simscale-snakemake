#!/bin/bash
#SBATCH --job-name=snakemake
#SBATCH --output=log/slurm/run_snakemake/slurm-%j.out
#SBATCH --error=log/slurm/run_snakemake/slurm-%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=10G
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=amjad_dabi@unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
eval "$(conda shell.bash hook)"
conda activate pyto

snakemake --profile profile/slurm "$@"