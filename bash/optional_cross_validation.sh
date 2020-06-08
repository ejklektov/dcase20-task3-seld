#!/bin/bash

# Data directory
DATASET_DIR='/home/ejklektov/dcase20-3/data/dcase20_seld_data/'

# Feature directory
FEATURE_DIR='/home/ejklektov/dcase20-3/data/dcase20_seld_data/features/'

# Workspace
WORKSPACE='/home/ejklektov/dcase20-3/surrey20/'
cd $WORKSPACE

########### Hyper-parameters ###########
FEATURE_TYPE='logmelintensity'  # 'logmel' | 'logmelgcc' | 'logmelintensity' | 'logmelgccintensity'
AUDIO_TYPE='foa'                # 'mic' | 'foa' | 'foa&mic'

# Batch size, max epochs, learning rate
BATCH_SIZE=32

# Chunk length
CHUNKLEN=5

# Model
MODEL_SED='CRNN11'              # 'CRNN11' | 'CRNN9' | 'Gated_CRNN9'
MODEL_DOA='pretrained_CRNN10'   # 'pretrained_CRNN10' | 'pretrained_CRNN8' | 'pretrained_Gated_CRNN8'

# Data augmentation
DATA_AUG='None'                 # 'None' | 'mixup' | 'specaug' | 'mixup&specaug'

# Name of the trial
NAME='BS32_5s'

# seed
SEED=30220

# GPU number
CUDA_VISIBLE_DEVICES=3

############ Development Evaluation ############
# test SED first
FUSION='True'                  # Ensemble or not: 'True' | 'False'
TASK_TYPE='sed_only'            # 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
for EPOCH_NUM in {38..40..2}
    do  
        echo $'\nEpoch numbers: '$EPOCH_NUM
        for FOLD in {1..4}
            do
            echo $'\nFold: '$FOLD
            python ${WORKSPACE}main/main.py test --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE \
            --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --batch_size=$BATCH_SIZE --model_sed=$MODEL_SED --model_doa=$MODEL_DOA --fold=$FOLD --epoch_num=$EPOCH_NUM \
            --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION --chunklen=$CHUNKLEN --CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
        done
        # mv ${WORKSPACE}appendixes/submissions/${NAME}_${MODEL_SED}_${AUDIO_TYPE}_${FEATURE_TYPE}_aug_${DATA_AUG}_seed_${SEED}/_fusion_sed \
        # ${WORKSPACE}appendixes/submissions/${NAME}_${MODEL_SED}_${AUDIO_TYPE}_${FEATURE_TYPE}_aug_${DATA_AUG}_seed_${SEED}/_fusion_sed_epoch_${EPOCH_NUM}
done

# ensemble sed on different iterations and write out probabilities
python ${WORKSPACE}main/ensemble.py iters_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# ensemble sed on different models and write out probabilities
python ${WORKSPACE}main/ensemble.py models_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# threshold the probabilities and write out submissions to 'sed_test_fusioned' folder
THRESHOLD=0.5
python ${WORKSPACE}main/ensemble.py threshold_models_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --threshold=$THRESHOLD

# test unfusioned sed
FUSION='False'
python ${WORKSPACE}main/main.py test_all --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION

# test fusioned sed
FUSION='True'
python ${WORKSPACE}main/main.py test_all --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION


# test DOA
FUSION='True'                  # Ensemble or not: 'True' | 'False'
TASK_TYPE='two_staged_eval'    # 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
for EPOCH_NUM in {78..80..2}
    do  
        echo $'\nEpoch numbers: '$EPOCH_NUM
        for FOLD in {1..4}
            do
            echo $'\nFold: '$FOLD
            python ${WORKSPACE}main/main.py test --workspace=$WORKSPACE --data_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE \
            --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --batch_size=$BATCH_SIZE --model_sed=$MODEL_SED --model_doa=$MODEL_DOA --fold=$FOLD --epoch_num=$EPOCH_NUM \
            --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION --chunklen=$CHUNKLEN --CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
        done
done

# ensemble doa
python ${WORKSPACE}main/ensemble.py iters_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# ensemble sed on different models and write out probabilities
python ${WORKSPACE}main/ensemble.py models_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME

# threshold the probabilities and write out submissions to 'sed_test_fusioned' folder
THRESHOLD=0.4
python ${WORKSPACE}main/ensemble.py threshold_models_ensemble --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --threshold=$THRESHOLD

# test unfusioned sed
FUSION='False'
python ${WORKSPACE}main/main.py test_all --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION

# test fusioned sed
FUSION='True'
python ${WORKSPACE}main/main.py test_all --workspace=$WORKSPACE --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE \
--model_sed=$MODEL_SED --model_doa=$MODEL_DOA --data_aug=$DATA_AUG --seed=$SEED --name=$NAME --fusion=$FUSION

####################################
