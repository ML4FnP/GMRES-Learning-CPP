#!/usr/bin/env bash

mkdir -p dist
pushd dist
cmake .. -DAMReX_ROOT=$HOME/Programs/amrex/dist
make VERBOSE=TRUE -j
cp main.ex ./..
popd
