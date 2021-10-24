mkdir ./experiments
mkdir ./experiments/trained_models/
mkdir ./experiments/trained_models/clf/

declare -a DataSets=("rosbank" "gender_tinkoff" "age_tinkoff" "gender" "age")
declare -a ModelTypes=("gru_with_amounts" "lstm_with_amounts" "cnn_with_amounts")

for dataset_name in ${DataSets[@]}; do
    mkdir ./experiments/trained_models/clf/${dataset_name}/
    mkdir ./presets/${dataset_name}/models/
    mkdir ./presets/${dataset_name}/models/clf
        for config_name in ${ModelTypes[@]}; do
    	    rm -rf experiments/trained_models/clf/${dataset_name}/${config_name}
    	    mkdir experiments/trained_models/clf/${dataset_name}/${config_name}
            bash scripts/local/train_clf.sh ${config_name} "100_quantile" ${dataset_name}
            rm -rf ./presets/${dataset_name}/models/${config_name}.tar.gz
            cp -r ./experiments/trained_models/clf/${dataset_name}/${config_name}/model.tar.gz ./presets/${dataset_name}/models/clf/${config_name}.tar.gz
        done
    done
