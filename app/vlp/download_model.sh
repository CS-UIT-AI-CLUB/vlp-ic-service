WORK_DIR=/usr/src
CHECKPOINTS=${WORK_DIR}/app/vlp/checkpoints
BERT=${WORK_DIR}/app/vlp

# Download BERT base multilingual cased model
mkdir -p $BERT
cd $BERT
gdown --id 1RQbUu_SUHC0M_Qu87TRxoYdknPGBnYep -O vlp-bert-base-multilingual-cased.zip
unzip vlp-bert-base-multilingual-cased.zip
rm vlp-bert-base-multilingual-cased.zip
mv bert-base-multilingual-cased bert

# Download checkpoints
mkdir -p $CHECKPOINTS
cd $CHECKPOINTS
gdown --id 19PyKSSlQI5mvtgWy21PyctgYElgcvfZE -O model.bin