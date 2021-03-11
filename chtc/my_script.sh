#!/bin/bash
echo "start running sh" 1>&2 
tar -xzf python36.tar.gz
#mkdir my_output
tar -xzf packages.tar.gz
ls /staging/xmiao27 1>&2 
#unzip /staging/xmiao27/land-train.zip
#ls -s > files.txt
unzip land-train.zip
echo "finished unzipping" 1>&2
export PATH=$PWD/python/bin:$PATH


export PYTHONPATH=$PWD/packages
export HOME=$PWD
#pip list 1>&2
python3 code.py
mv *.npy /staging/xmiao27
rm -r land-train/*
