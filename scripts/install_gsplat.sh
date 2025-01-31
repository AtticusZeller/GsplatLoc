#! /usr/bin/env bash

cd ..
pip uninstall gsplat
cd ./thirdparty/gsplat
#rm -rf build
pip install .
