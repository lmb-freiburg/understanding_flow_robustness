#!/bin/bash

echo "Installer for PyTorch:"
echo "Which cuda version do you use [9.2, 10.0 (we used this), 10.1, cpu]?"

read version

echo $version

if [ $version == "9.2" ]
then
    echo "Install PyTorch 1.4.0 for CUDA 9.2"
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=9.2 -c pytorch
elif [ $version == "10.0" ]
then
    echo "Install PyTorch 1.4.0 for CUDA 10.0"
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
elif [ $version == "10.1" ]
then
    echo "Install PyTorch 1.4.0 for CUDA 10.1"
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
elif [ $version == "cpu" ]
then
    echo "Install PyTorch 1.4.0 for cpu"
    conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
else
    echo "Version undefinied. You might need to adjust this script!"
fi
