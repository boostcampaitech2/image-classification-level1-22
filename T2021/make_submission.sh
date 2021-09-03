#!/bin/bash

python train.py --epochs=1

cd ./checkpoint || exit
fn=$(ls -t | head -n1)

cd ../ || exit

python inference.py --pt_name=$fn --submission_dir='./'