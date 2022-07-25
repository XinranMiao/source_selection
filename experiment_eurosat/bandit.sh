#!/bin/bash
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD
source activate example-environment

git clone https://github.com/XinranMiao/source_selection
cd source_selection/experiment_eurosat
cp ~/metadata.csv .

python3 bandit_run.py $1 $2
