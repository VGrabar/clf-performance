

declare -a DataSets=("rosbank" "gender_tinkoff" "gender" "age_tinkoff") 
declare -a ModelTypes=("gru_with_amounts" "lstm_with_amounts" "cnn_with_amounts")
#declare -a ModelTypes=("gru_with_amounts")

for dataset_name in ${DataSets[@]}; do
    DATA_PATH=./data/${dataset_name}/test
    for config_name in ${ModelTypes[@]}; do
        SAVE_PATH=./data/${dataset_name}/plot_${config_name}_${dataset_name}.json
        MODEL_PATH=./presets/${dataset_name}/models/clf/${config_name}.tar.gz
        PYTHONPATH=. python scripts/python_scripts/run_inference.py ${DATA_PATH} ${MODEL_PATH} ${SAVE_PATH}
        done
    done