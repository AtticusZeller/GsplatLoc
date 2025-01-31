#!/usr/bin/env bash

mkdir -p Datasets
wget https://huggingface.co/datasets/Atticux/GsplatLoc/resolve/main/data.zip?download=true
unzip dataset.zip -d Datasets
