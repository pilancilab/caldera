#!/bin/bash
git submodule init
git submodule update
cp quip-sharp-setup.py quip-sharp/setup.py
cd quip-sharp && pip install --editable . && cd ..
cd quip-sharp/quiptools && python setup.py install && cd ../..

