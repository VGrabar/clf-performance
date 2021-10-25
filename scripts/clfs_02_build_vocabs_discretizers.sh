#!/usr/bin/env bash

#declare -a DataSets=("age" "gender" "gender_tinkoff" "age_tinkoff" "rosbank")
declare -a DataSets=("gender_tinkoff" "age_tinkoff" "rosbank" "gender")

for dataset_name in ${DataSets[@]}; do
	rm -rf ./presets/${dataset_name}/vocabs/100_quantile
	rm -rf ./presets/${dataset_name}/discretizers/100_quantile
	rm -rf ./presets/${dataset_name}/discretizers/50_quantile
	mkdir presets/${dataset_name}/
	mkdir presets/${dataset_name}/discretizers
	mkdir presets/${dataset_name}/vocabs

	PYTHONPATH=. python scripts/python_scripts/train_discretizers.py ${dataset_name}
	PYTHONPATH=. python scripts/python_scripts/build_vocabs.py ${dataset_name}
done

