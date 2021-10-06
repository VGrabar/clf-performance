#!/usr/bin/env bash

CONFIG_NAME=${1:-"gru_with_amounts"}
DISCRETIZER_NAME=${2:-"100_quantile"}
DATASET_NAME=${3:-"age"}

rm -rf ./experiments/trained_models/${DATASET_NAME}/clf/${CONFIG_NAME}

CLF_TRAIN_DATA_PATH=./data/${DATASET_NAME}/train.jsonl \
CLF_VALID_DATA_PATH=./data/${DATASET_NAME}/valid.jsonl \
CLF_TEST_DATA_PATH=./data/${DATASET_NAME}/test.jsonl \
DISCRETIZER_PATH=./presets/${DATASET_NAME}/discretizers/${DISCRETIZER_NAME} \
VOCAB_PATH=./presets/${DATASET_NAME}/vocabs/${DISCRETIZER_NAME} \
RANDOM_SEED=0 \
allennlp train ./configs/classifiers/${CONFIG_NAME}.jsonnet \
--serialization-dir ./experiments/trained_models/${DATASET_NAME}/clf/${CONFIG_NAME} \
--include-package advsber
