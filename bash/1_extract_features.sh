#!/bin/bash

# Data directory
DATASET_DIR='/home/ejklektov/dcase20-3/data/dcase20_seld_data/'

# Feature directory
	#FEATURE_DIR='/vol/vssp/msos/YinC/workspace/Dataset_Features/DCASE2019/task3/'
FEATURE_DIR='/home/ejklektov/dcase20-3/data/dcase20_seld_data/features/'

# Workspace
WORKSPACE='/home/ejklektov/dcase20-3/surrey20/'
cd $WORKSPACE

########### Hyper-parameters ###########
FEATURE_TYPE='logmelgccintensity'  # 'logmel' | 'logmelgcc' | 'logmelintensity' | 'logmelgccintensity'
AUDIO_TYPE='foa&mic'                # 'mic' | 'foa' | 'foa&mic'

############ Extract Features ############
# dev
python3 utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='dev' --audio_type=$AUDIO_TYPE

# eval
python3 utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='eval' --audio_type=$AUDIO_TYPE
