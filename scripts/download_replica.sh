#! /usr/bin/env bash

mkdir -p Datasets
cd Datasets || exit
wget https://github.com/SupaVision/AutoDrive_frontend/releases/download/Dataset/Replica.zip
unzip Replica.zip
rm -rf Replica.zip
