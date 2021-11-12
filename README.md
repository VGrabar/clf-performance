# Out-of-sample performance of classification models


## Usage
### Step 0. Install dependencies

```bash
poetry install
poetry shell
```

## Reproducibility

To reproduce all  experiments, please, run all bash scripts from `./scripts` in numerical order:

```
clfs_01_build_datasets.sh
clfs_02_build_vocabs_discretizers.sh
clfs_03_train_all_classifiers.sh
clfs_04_run_inference.sh
```

### Step 1. Building datasets

We are working with following transactional datasets: Age, Age (Tinkoff), Gender, Gender (Tinkoff) and Rosbank

To get the processed datasets, you need to run

`bash scripts/clfs_01_build_datasets.sh`

### Step 2. Building vocabs and discretizers.
To build vocabulary and train discretizer run:

`bash scripts/clfs_02_build_vocabs_discretizers.sh`

Trained discretizers will be stored in `./presets/${dataset_name}/discretizers/100_quantile`, and vocabs in `./presets/${dataset_name}/vocabs/100_quantile`.

## Experiments

All results will be at `./experiments`:

1. Trained models: `./experiments/trained_models`

### Step 3. Training all classifiers.

To train all classifiers (LSTM, CNN, GRU) run:

`bash scripts/clfs_03_train_all_classifiers.sh`

As a result, all trained models will be stored in `./experiments/trained_models`.


### Step 4. Running inference on full dataset

To check model performance please run:

`bash scripts/clfs_04_run_inference.sh`

The results (metrics saved by period) for each dataset and each classifier will be stored in `./data/${dataset_name}/plot_${config_name}_${dataset_name}.json`. 
