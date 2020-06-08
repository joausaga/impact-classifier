#!/bin/bash
#SBATCH--job-name="social-impact-research-model-training"
#SBATCH--workdir=.
#SBATCH--output=train_rf_%j.out
#SBATCH--error=train_rf_%j.err
#SBATCH--ntasks=1
#SBATCH--time=24:00:00
#SBATCH--cpus-per-task=24

echo $(date '+%Y-%m-%d %H:%M:%S')
module purge
module load intel
module load mkl
module load python/3.7.4
source env/bin/activate
python train.py train-model RF './data/train/data_tfidf_10000_1_ 2_clean_lemmatization.pkl' --algorithm_name 'random_forest' --metric 'balanced_accuracy'
echo $(date '+%Y-%m-%d %H:%M:%S')