#!/bin/bash
#SBATCH--job-name="social-impact-research-model-training"
#SBATCH--workdir=.
#SBATCH--output=train_lr_%j.out
#SBATCH--error=train_lr_%j.err
#SBATCH--ntasks=1
#SBATCH--time=24:00:00
#SBATCH--cpus-per-task=24

echo $(date '+%Y-%m-%d %H:%M:%S')
module purge
module load intel
module load mkl
module load python/3.7.4
source env/bin/activate
python train.py train-model LR './data/train/data_tc_1000_1_ 2_clean.pkl' --algorithm_name 'logistic_regression' --metric 'f1'
echo $(date '+%Y-%m-%d %H:%M:%S')