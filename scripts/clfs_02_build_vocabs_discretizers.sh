#!/usr/bin/env bash

rm -rf ./presets/rosbank/vocabs/100_quantile
rm -rf ./presets/rosbank/discretizers/100_quantile
rm -rf ./presets/rosbank/discretizers/50_quantile
mkdir presets/rosbank/
mkdir presets/rosbank/discretizers
mkdir presets/rosbank/vocabs

PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'rosbank'
PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'rosbank'

