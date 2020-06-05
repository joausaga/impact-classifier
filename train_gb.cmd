#!/bin/bash
#SBATCH--job-name="social-impact-research-model-training"
#SBATCH--workdir=.
#SBATCH--output=train_gb_%j.out
#SBATCH--error=train_gb_%j.err
#SBATCH--ntasks=1
#SBATCH--time=24:00:00
#SBATCH--cpus-per-task=24

echo $(date '+%Y-%m-%d %H:%M:%S')
module purge
module load intel
module load mkl
module load python/3.7.4
source env/bin/activate
python train.py train-model GB './data/train_test/data_tfidf_10000_(1, 3)_clean_lemmatization.pkl' --algorithm_name 'gradient_boosting' --metric 'balanced_accuracy'
echo $(date '+%Y-%m-%d %H:%M:%S')