#! /usr/bin/env bash

cd ..
pip uninstall small_gicp
cd ./thirdparty/small_gicp
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
cd ..
pip install .
