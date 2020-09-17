# """
# ClovaCall
# Copyright 2020-present NAVER Corp.
# MIT license
# """
#!/bin/bash

MAIN_DIR=${0%/*}
cd $MAIN_DIR/..

TARGET_CODE=las.pytorch/main.py
MODEL_PATH=models
LOG_PARENT_PATH=log

if [ ! -f $TARGET_CODE ]; then
    echo "[ERROR] TARGET_CODE($TARGET_CODE) not found."
    exit
fi

if [ ! -d $MODEL_PATH ]; then
    mkdir $MODEL_PATH
fi

if [ ! -d $LOG_PARENT_PATH ]; then
    mkdir $LOG_PARENT_PATH
fi

################################################################
##	Careful while modifying lines above.
################################################################

TRAIN_FILE=data/zeroth_korean/train_zeroth_korean.trimmed.json
TEST_FILE=data/zeroth_korean/test_zeroth_korean.trimmed.json
LABEL_FILE=data/kor_syllable_zeroth.json
DATASET_PATH=data/zeroth_korean

CUDA_DEVICE_ID=1 # 0

# Default
RNN_TYPE=LSTM
BATCH_SIZE=16 # 32
LR=3e-4
LR_ANNEAL=1.1
DROPOUT=0.3
TF_RATIO=1.0
EPOCHS=100

# LAS
ENCODER_LAYERS=3
ENCODER_SIZE=512
DECODER_LAYERS=2
DECODER_SIZE=512

GPU_SIZE=1
CPU_SIZE=8 # 4 

MAX_LEN=128

TRAIN_INFO="zeroth_korean_trimmed_uttmvn" # "ClovaCall"
MODE="test"


################################################################
## Experiment
################################################################
norm_type="utterance_mvn"

if [ $norm_type = "instance" ]; then
    CUDA_DEVICE_ID=1
    TRAIN_INFO="zeroth_korean_trimmed_instance+inputnorm"
elif [ $norm_type = "utterance_mvn" ]; then
    # CUDA_DEVICE_ID=0 
    # TRAIN_INFO="zeroth_korean_trimmed_utterance_mvn+inputnorm+cnn+espnetfbank"
    CUDA_DEVICE_ID=1 
    TRAIN_INFO="zeroth_korean_trimmed_utterance_mvn+inputnorm+cnn+kaldifbank"
else
    exit 1
fi


################################################################
##	Careful while modifying lines below.
################################################################

MODELS_PATH=models/$TRAIN_INFO


CUR_MODEL_PATH=${MODELS_PATH}/${RNN_TYPE}_${ENCODER_SIZE}x${ENCODER_LAYERS}_${DECODER_SIZE}x${DECODER_LAYERS}_${TRAIN_INFO}
LOG_CHILD_PATH=${LOG_PARENT_PATH}/${RNN_TYPE}_${ENCODER_SIZE}x${ENCODER_LAYERS}_${DECODER_SIZE}x${DECODER_LAYERS}_${TRAIN_INFO}

LOG_FILE=$LOG_CHILD_PATH/run_las_asr_decode_CUDA${CUDA_DEVICE_ID}.sh.log

if [ ! -d $MODELS_PATH ]; then
    mkdir $MODELS_PATH
fi

if [ ! -d $CUR_MODEL_PATH ]; then
    mkdir $CUR_MODEL_PATH
fi

if [ ! -d $LOG_CHILD_PATH ]; then
    mkdir $LOG_CHILD_PATH
fi


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID \
python -u $TARGET_CODE \
--batch_size $BATCH_SIZE \
--num_workers $CPU_SIZE \
--num_gpu $GPU_SIZE \
--rnn-type $RNN_TYPE \
--lr $LR \
--learning-anneal $LR_ANNEAL \
--dropout $DROPOUT \
--teacher_forcing $TF_RATIO \
--encoder_layers $ENCODER_LAYERS --encoder_size $ENCODER_SIZE \
--decoder_layers $DECODER_LAYERS --decoder_size $DECODER_SIZE \
--train-file $TRAIN_FILE --test-file-list $TEST_FILE \
--labels-path $LABEL_FILE \
--dataset-path $DATASET_PATH \
--load-model --mode $MODE \
--max_len $MAX_LEN \
--cuda --save-folder $CUR_MODEL_PATH --model-path $CUR_MODEL_PATH/final.pth --log-path $LOG_CHILD_PATH | tee $LOG_FILE 
