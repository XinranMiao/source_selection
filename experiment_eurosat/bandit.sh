#!/bin/bash
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD
source activate example-environment

git clone https://github.com/XinranMiao/source_selection
cd source_selection/experiment_eurosat
cp ~/metadata_clustered.csv .

python3 bandit_run.py $1 $2 $3

tar -czf derived_data_"$1"_"$2"_"$3".tar.gz derived_data
#cp derived_data_"$1"_"$2".tar.gz /staging/xmiao27
mv derived_data_"$1"_"$2"_"$3".tar.gz ~
