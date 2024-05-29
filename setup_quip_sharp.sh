#!/bin/bash
git submodule init
git submodule update
cp quip-sharp-pyproject.toml quip-sharp/pyproject.toml
cd quip-sharp && pip install --editable . && cd ..
cd quip-sharp/quiptools && python setup.py install && cd ../..

