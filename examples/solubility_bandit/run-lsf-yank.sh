#!/bin/bash
#  Batch script for mpirun job on cbio LSF cluster
#  Adjust your script as needed for your clusters!
#
# walltime : maximum wall clock time (hh:mm)
#BSUB -W 72:00
#
# Set Output file
#BSUB -o  log.%J.log
#
# Specify node group
#BSUB -m ls-gpu 
#
# Specify the correct queue to use GPU's
#BSUB -q gpuqueue
#
# 4 CPU and 4 GPU on 1 node
#BSUB -n 4 -R "rusage[mem=8] span[ptile=4]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# Start MPS since Cbio GPUs are in exclusive mode
#BSUB -env "all, LSB_START_JOB_MPS=Y"
#
# job name (default = name of script file)
#BSUB -J "benzeneset"

# Run the simulation with verbose output:
echo "Running simulation via MPI..."
echo $PWD
echo "Running in ^^^"
python solubility.py 
date
