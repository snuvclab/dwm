#!/bin/bash
#SBATCH --job-name=run_egoallo_aria
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --comment="running egoallo on aria dataset for pseudo gt human motion"
#SBATCH --output=out/run_egoallo_aria_%j.out
#SBATCH --error=out/run_egoallo_aria_%j.err

cmd="bash data_processing/egoallo/run_egoallo.sh /virtual_lab/jhb_vclab/world_model/data/aria/aria_digital_twins/Apartment_release_clean_seq131_M1292"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"