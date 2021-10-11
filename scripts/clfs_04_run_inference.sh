

declare -a DataSets=("rosbank")
declare -a ModelTypes=("gru_with_amounts" "lstm_with_amounts" "cnn_with_amounts")
#declare -a ModelTypes=("gru_with_amounts")

for dataset_name in ${DataSets[@]}; do
    DATA_PATH=./data/${dataset_name}/test
    for config_name in ${ModelTypes[@]}; do
        SAVE_PATH=./data/${dataset_name}/plot_${config_name}.json
        MODEL_PATH=./presets/${dataset_name}/models/clf/${config_name}.tar.gz
        PYTHONPATH=. python advsber/commands/run_inference.py ${DATA_PATH} ${MODEL_PATH} ${SAVE_PATH}
        done
    done