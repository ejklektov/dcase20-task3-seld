#!/bin/bash

# Data directory
DATASET_DIR='/home/ejklektov/dcase20-3/data/dcase20_seld_data/'

# Feature directory
FEATURE_DIR='/home/ejklektov/dcase20-3/data/dcase20_seld_data/features/'

# Workspace
WORKSPACE='/home/ejklektov/dcase20-3/surrey20/'
cd $WORKSPACE

########### Hyper-parameters ###########
FEATURE_TYPE='logmelgccintensity'   # 'logmel' | 'logmelgcc' | 'logmelintensity' | 'logmelgccintensity'
AUDIO_TYPE='foa&mic'                # 'mic' | 'foa' | 'foa&mic'
FOLD=1                              # 1-4 for folds in dev set, -1 for eval sets

# Batch size, max epochs, learning rate
BATCH_SIZE=32

# Chunk length
CHUNKLEN=5

# Model
MODEL_SED='CRNN9_logmelgccintensity'              # 'CRNN11' | 'CRNN9' | 'Gated_CRNN9' | 'CRNN9_logmelgccintensity' | 'CRNN11_logmelgccintensity'
MODEL_DOA='pretrained_CRNN8_logmelgccintensity'   # 'pretrained_CRNN10' | 'pretrained_CRNN8' | 'pretrained_Gated_CRNN8' | 'pretrained_CRNN8_logmelgccintensity' | 'pretrained_CRNN10_logmelgccintensity'
PRETRAINED_EPOCH=40             # pretrained epoch for DOA

# Data augmentation
DATA_AUG='mixup&specaug'                 # 'None' | 'mixup' | 'specaug' | 'mixup&specaug'

# Learning rate
LR=1e-3
REDUCE_LR='True'

# Name of the trial
NAME='BS32_5s' # 'n0' | 'test'

# seed
SEED=30250

# GPU number
CUDA_VISIBLE_DEVICES="6,7"

############ Development ############
## train
# SED
TASK_TYPE='sed_only'            # 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
MAX_EPOCHS=40
RESUME_EPOCH=-1                  # resume training, -1 for train from scratch, positive integer for resuming the epoch from

python3 ${WORKSPACE}main/main.py train --workspace=$WORKSPACE --data_dir $DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE \
--audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --batch_size=$BATCH_SIZE --max_epochs=$MAX_EPOCHS --lr=$LR --reduce_lr=$REDUCE_LR --model_sed=$MODEL_SED \
--model_doa=$MODEL_DOA --fold=$FOLD --pretrained_epoch=$PRETRAINED_EPOCH --resume_epoch=$RESUME_EPOCH --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --chunklen=$CHUNKLEN \
--CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# DOA
TASK_TYPE='doa_only'            # 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
MAX_EPOCHS=80
RESUME_EPOCH=-1                  # resume training, -1 for train from scratch, positive integer for resuming the epoch from

python3 ${WORKSPACE}main/main.py train --workspace=$WORKSPACE --data_dir $DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE \
--audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --batch_size=$BATCH_SIZE --max_epochs=$MAX_EPOCHS --lr=$LR --reduce_lr=$REDUCE_LR --model_sed=$MODEL_SED \
--model_doa=$MODEL_DOA --fold=$FOLD --pretrained_epoch=$PRETRAINED_EPOCH --resume_epoch=$RESUME_EPOCH --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --chunklen=$CHUNKLEN \
--CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

