#!/usr/bin/env bash

RATIO="30"
# rosbank
DATASET_NAME="rosbank"
rm -rf data/$DATASET_NAME
mkdir data/$DATASET_NAME
mkdir data/$DATASET_NAME/original

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gVprY6E6jK_VZHFxkOXSuLjPqDTYog7t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gVprY6E6jK_VZHFxkOXSuLjPqDTYog7t" -O 'data/rosbank/original/train.csv' && rm -rf /tmp/cookies.txt


PYTHONPATH=. python scripts/python_scripts/create_rosbank.py $DATASET_NAME $RATIO
