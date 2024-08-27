#!/bin/bash
set -e
cur_dir=$(cd $(dirname $0) && pwd)

if [ -n $CONDA_VOLUME ]; then
    export CONDA_VOLUME=/home/heenj/anaconda3
fi

CONDA_ENV=jupyter
source $CONDA_VOLUME/etc/profile.d/conda.sh


# if conda env list | grep -q $CONDA_ENV; then
#     echo "Conda environment does not exist.."
#     echo "Creates Conda env..."
# fi

conda activate $CONDA_ENV


set -a
source $cur_dir/conf/credentials
set +a
# bash $cur_dir/src/sql/create_tables.sh

if [ -z $1 ]; then
    echo "You should enter mode type of full_save."
    exit 1
fi


echo "Current time is `date`."
export _ts=`date +"%Y-%m-%d %H:%M:%S %z"`
export full_save=$1


echo "Run dart module!"
python3 $cur_dir/main.py

echo "Finish time is `date`."
conda deactivate


