#!/usr/bin/env bash

RATIO="30"
# rosbank
#rm -rf data/rosbank
#mkdir data/rosbank
#mkdir data/rosbank/original

#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gVprY6E6jK_VZHFxkOXSuLjPqDTYog7t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gVprY6E6jK_VZHFxkOXSuLjPqDTYog7t" -O 'data/rosbank/original/train.csv' && rm -rf /tmp/cookies.txt

#PYTHONPATH=. python scripts/python_scripts/create_rosbank.py $RATIO
# gender
#rm -rf ./data/gender
#mkdir ./data/gender
#mkdir ./data/gender/original

#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bxGlxX9-T1vEc-x-AU1OWB4H6po2jWrm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bxGlxX9-T1vEc-x-AU1OWB4H6po2jWrm" -O './data/gender/original/transactions.csv' && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IlObdLeQkr5mZP0iVv2WcJTsxz4Nz62h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IlObdLeQkr5mZP0iVv2WcJTsxz4Nz62h" -O './data/gender/original/gender_train.csv' && rm -rf /tmp/cookies.txt

#PYTHONPATH=. python scripts/python_scripts/create_gender.py $RATIO
# age
# DATASET_NAME="age"
# rm -rf data/$DATASET_NAME
# mkdir data/$DATASET_NAME
# mkdir data/$DATASET_NAME/original

# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1roUn50LtDDPcl3UsI_1phZh-EryVstZi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1roUn50LtDDPcl3UsI_1phZh-EryVstZi" -O './data/age/original/transactions_train.csv' && rm -rf /tmp/cookies.txt

# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QiNIOfoJhz0ILHJ3Xk_gunE7yPVMSI-8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QiNIOfoJhz0ILHJ3Xk_gunE7yPVMSI-8" -O './data/age/original/train_target.csv' && rm -rf /tmp/cookies.txt


# PYTHONPATH=. python scripts/python_scripts/create_age.py $RATIO

mkdir ./data/age_tinkoff
mkdir ./data/age_tinkoff/original

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB" -O './data/age_tinkoff/original/transactions.csv' && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ" -O './data/age_tinkoff/original/customer_train.csv' && rm -rf /tmp/cookies.txt


PYTHONPATH=. python scripts/python_scripts/create_age_tinkoff.py $RATIO


mkdir ./data/gender_tinkoff
mkdir ./data/gender_tinkoff/original

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB" -O './data/gender_tinkoff/original/transactions.csv' && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ" -O './data/gender_tinkoff/original/customer_train.csv' && rm -rf /tmp/cookies.txt


PYTHONPATH=. python scripts/python_scripts/create_gender_tinkoff.py $RATIO

