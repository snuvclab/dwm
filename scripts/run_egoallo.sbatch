#!/bin/bash
#SBATCH --job-name=run_egoallo_aria
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --comment="running egoallo on aria dataset for pseudo gt human motion"
#SBATCH --output=out/run_egoallo_aria_%j.out
#SBATCH --error=out/run_egoallo_aria_%j.err

folder="$1"
echo "Running egoallo on folder: $folder"

bash data_processing/egoallo/run_egoallo.sh "$folder"
exit_code=$?

echo -ne "-------------------- Finished executing SLURM job --------------------\n\n"
exit $exit_code