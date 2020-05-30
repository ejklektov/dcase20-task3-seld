import os
import pdb
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix

from loss import hybrid_regr_loss
from metrics import cls_feature_class, evaluation_metrics, SELD_evaluation_metrics
from utils.utilities import to_np, to_torch


def evaluate(data_generator, data_type, max_audio_num, task_type, model, cuda, loss_type, 
        threshold, data_dir, submissions_dir=None, frames_per_1s=100, sub_frames_per_1s=50, FUSION=False, epoch_num=None, max_epochs=None, cur_epoch=None):
    '''
    Evaluate metrics for cross validation or test data

    Input:
        data_generator: data loader
        data_type: 'train' | 'valid' | 'test'
        max_audio_num: maximum audio number to evaluate the performance, None for using all clips
        task_type: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
        model: nn model
        cuda: True or False to use cuda or not
        threshold: {'sed': event detection threshold,
                    'doa': doa threshold}
    Returns:
        loss_dict: {'loss': event_loss + beta*(elevation_loss + azimuth_loss),
                    'event_loss': event_loss,
                    'doa_loss': elevation_loss + azimuth_loss}
    '''

    if data_type == 'train':
        generate_func = data_generator.generate_test(data_type='train', 
            max_audio_num=max_audio_num)
    elif data_type == 'valid':
        generate_func = data_generator.generate_test(data_type='valid', 
            max_audio_num=max_audio_num)
    elif data_type == 'test':
        generate_func = data_generator.generate_test(data_type='test',
            max_audio_num=max_audio_num)

    sed_gt = []
    doa_gt = []
    sed_pred = []
    doa_pred = []

    forLoss_sed_pred = []
    forLoss_doa_pred = []

    for batch_x, batch_y_dict, batch_fn in generate_func:
        '''
        batch_size = 1
        batch_x: features
        batch_y_dict = {
            'events',       (time_len, class_num)
            'doas',         (time_len, 2*class_num) for 'regr' | 
                            (time_len, class_num, ele_num*azi_num=324) for 'clas'
            # 'distances'     (time_len, class_num)
        }
        batch_fn: filenames
        '''
        batch_x = to_torch(batch_x, cuda)
        with torch.no_grad():
            model.eval()
            output = model(batch_x)
        output['events'] = to_np(output['events'])
        output['doas'] = to_np(output['doas'])
        '''
        output = {
            'events',   (batch_size=1, time_len, class_num) 
            'doas'      (batch_size=1, time_len, 2*class_num) for 'regr' | 
                        (batch_size=1, time_len, ele_num*azi_num=324) for 'clas'
        }
        '''

        #############################################################################################################
        # save predicted sed results in 'sed_only' task
        # set output['events'] to ground truth sed in 'doa_only' task
        # load predicted sed results in 'two_staged_eval' task
        sed_mask_dir = os.path.join(os.path.abspath(os.path.join(submissions_dir, os.pardir)), '_sed_mask')
        os.makedirs(sed_mask_dir, exist_ok=True)
        hdf5_path = os.path.join(sed_mask_dir, batch_fn + '.h5')
        if task_type == 'sed_only':
            with h5py.File(hdf5_path, 'w') as hf:
                hf.create_dataset('sed_pred', data=output['events'], dtype=np.float32)
        elif task_type == 'doa_only':
            ###set predictions is equal to ground truth
            temp = np.expand_dims(batch_y_dict['events'], axis=0)
            if output['events'].shape[1] <= temp.shape[1]:
                output['events'] = temp[:, 0: output['events'].shape[1]]
            else:
                output['events'] = np.concatenate((temp, 
                        np.zeros((1, output['events'].shape[1]-temp.shape[1], temp.shape[2]))), axis=1)
        elif task_type == 'two_staged_eval':
            with h5py.File(hdf5_path, 'r') as hf:
                output['events'] = hf['sed_pred'][:]
        #############################################################################################################

        min_idx = min(batch_y_dict['events'].shape[0], output['events'].shape[1])

        sed_gt.append(batch_y_dict['events'][:min_idx])
        doa_gt.append(batch_y_dict['doas'][:min_idx])

        sed_pred.append(output['events'].squeeze()[:min_idx])
        doa_pred.append(output['doas'].squeeze()[:min_idx])

        forLoss_sed_pred.append(output['events'].squeeze()[:min_idx])
        forLoss_doa_pred.append(output['doas'].squeeze()[:min_idx])

        ########################################## Interpolation ########################################
        output_events= interp_tensor(output['events'].squeeze(), frames_per_1s, sub_frames_per_1s)
        output_doas = interp_tensor(output['doas'].squeeze(), frames_per_1s, sub_frames_per_1s)
        gt_events= interp_tensor(batch_y_dict['events'], frames_per_1s, sub_frames_per_1s)
        #output_events= output['events'].squeeze()
        #output_doas = output['doas'].squeeze()
        #gt_events= batch_y_dict['events']
        ###########################################################################################  ######

        ################## Write probability and ground_truth to csv file for fusion ####################
        if FUSION and task_type == 'sed_only':
            fn_prob = '{}_prob.csv'.format(batch_fn)
            fn_gt = '{}_gt.csv'.format(batch_fn)
            fusion_sed_dir = os.path.join(os.path.abspath(os.path.join(submissions_dir, os.pardir)), '_fusion_sed_epoch_{}'.format(epoch_num))
            os.makedirs(fusion_sed_dir, exist_ok=True)
            gt_sed_dir = os.path.join(os.path.abspath(os.path.join(submissions_dir, os.pardir)), 'sed_mask_gt')
            os.makedirs(gt_sed_dir, exist_ok=True)
            file_path_prob = os.path.join(fusion_sed_dir, fn_prob)
            file_path_gt = os.path.join(gt_sed_dir, fn_gt)
        
            df_output = pd.DataFrame(output_events)
            df_output.to_csv(file_path_prob)
            df_gt = pd.DataFrame(gt_events)
            df_gt.to_csv(file_path_gt)

        elif FUSION and task_type == 'two_staged_eval':
            fn_doa = '{}_doa.csv'.format(batch_fn)
            fusion_doa_dir = os.path.join(os.path.abspath(os.path.join(submissions_dir, os.pardir)), '_fusion_doa_epoch_{}'.format(epoch_num))
            os.makedirs(fusion_doa_dir, exist_ok=True)
            file_path_doa = os.path.join(fusion_doa_dir, fn_doa)

            df_output = pd.DataFrame(output_doas)
            df_output.to_csv(file_path_doa)
        #################################################################################################
        
        ############################### for submission method evaluation ################################
        if FUSION and task_type == 'two_staged_eval':
            submission_models_dir = os.path.abspath(os.path.join(submissions_dir, os.pardir))
            fusion_dir = os.path.join(os.path.abspath(os.path.join(submission_models_dir, os.pardir)), 'models_ensemble', \
                'sed_mask_models_fusioned')

            # fusion_dir = os.path.join(os.path.abspath(os.path.join(submissions_dir, os.pardir)), 'sed_mask_fusioned')
            fn_path = os.path.join(fusion_dir, batch_fn+'_prob.csv')
            prob_fusioned = pd.read_csv(fn_path, header=0, index_col=0).values
            submit_dict = {
                'filename': batch_fn,
                'events': (prob_fusioned>threshold['sed']).astype(np.float32),
                'doas': output_doas
            }
        else:
            submit_dict = {
                'filename': batch_fn,
                'events': (output_events>threshold['sed']).astype(np.float32),
                'doas': output_doas
            }
        write_submission(submit_dict, submissions_dir)
        #################################################################################################

    sed_gt = np.concatenate(sed_gt, axis=0)
    doa_gt = np.concatenate(doa_gt, axis=0)
    sed_pred = np.concatenate(sed_pred, axis=0)
    doa_pred = np.concatenate(doa_pred, axis=0)

    ###################### SED and DOA metrics, for submission method evaluation ######################
    gt_meta_dir = os.path.join(data_dir, 'dev', 'metadata_dev')
    sed_scores19, doa_er_metric19, seld_metric19 = calculate_SELD_metrics(gt_meta_dir, submissions_dir, score_type='all')

    param20 = {
        'best_seld_metric' : 99999,
        'best_epoch' : -1,
        'patience_cnt' : 0,
        'new_metric' : np.zeros(4),
        'new_seld_metric' : np.zeros(1),
        'frames_per_1s' : frames_per_1s
    }
    new_metric, new_seld_metric = calculate_SELD_metrics20(sed_pred, sed_gt, doa_pred, doa_gt, param20, cur_epoch)
    ###################################################################################################

    ## mAP
    sed_mAP_micro = average_precision_score(sed_gt, sed_pred, average='micro')
    sed_mAP_macro = average_precision_score(sed_gt, sed_pred, average='macro')
    sed_mAP = [sed_mAP_micro, sed_mAP_macro]

    ## loss
    forLoss_gt_dict = {
        'events': to_torch(sed_gt[None,:,:], cuda=False),
        'doas':   to_torch(doa_gt[None,:,:], cuda=False)
    }
    forLoss_pred_dict = {
        'events': to_torch(np.concatenate(forLoss_sed_pred, axis=0)[None,:,:], cuda=False),
        'doas':   to_torch(np.concatenate(forLoss_doa_pred, axis=0)[None,:,:], cuda=False)
    }

    seld_loss, sed_loss, doa_loss = hybrid_regr_loss(forLoss_pred_dict, forLoss_gt_dict, task_type, loss_type=loss_type)
    loss = [to_np(seld_loss), to_np(sed_loss), to_np(doa_loss)]

    metrics19 = [loss, sed_mAP, sed_scores19, doa_er_metric19, seld_metric19, new_metric, new_seld_metric]

    # import neptune
    # neptune.log_metric('metric19/loss', loss)
    # neptune.log_metric('metric19/sed_mAP_micro', sed_mAP[0])seld_loss
    # neptune.log_metric('metric19/sed_mAP_macro', sed_mAP[1])
    # neptune.log_metric('metric19/sed_scores19', sed_scores19)
    # neptune.log_metric('metric19/doa_er_metric19', doa_er_metric19)
    # neptune.log_metric('metric19/seld_metric19', seld_metric19)
    #
    # neptune.log_metric('metric19/seld_metric19', new_metric)
    # neptune.log_metric('metric19/seld_metric19', new_metric)
    # neptune.log_metric('metric19/seld_metric19', new_seld_metric)
    # neptune.log_metric('metric19/seld_metric19', seld_metric19)

    # torch.cuda.empty_cache()

    return metrics19


def calculate_submission(output_dict, frames_per_1s, sub_frames_per_1s=50):
    '''
    Interoplate tensor to length of 20ms
    '''
    
    output_dict['events'] = interp_tensor(output_dict['events'], frames_per_1s, sub_frames_per_1s)
    output_dict['doas'] = interp_tensor(output_dict['doas'], frames_per_1s, sub_frames_per_1s)

    return output_dict


def interp_tensor(tensor, frames_per_1s, sub_frames_per_1s=50):
    '''
    Interpolate tensor
    
    Args:
        tensor: (time_steps, event_class_num)
        frames_per_1s: submission frames_per_1s
        sub_frames_per_1s: submission frames per 1 s
    '''
    ratio = 1.0 * sub_frames_per_1s / frames_per_1s

    new_len = int(np.around(ratio * tensor.shape[0]))
    new_tensor = np.zeros((new_len, tensor.shape[1]))

    for n in range(new_len):
        new_tensor[n] = tensor[int(np.around(n / ratio))]
    
    return new_tensor


def write_submission(dict, submissions_dir):
    '''
    Write predicted result to submission csv files

    Args:
        dict={
            'filename': file name,
            'events': (time_len, class_num)
            'doas': (time_len, 2*class_num) for 'regr' | 
                    (time_len, ele_num*azi_num=324) for 'clas'
        }
    '''

    fn = '{}.csv'.format(dict['filename'])
    file_path = os.path.join(submissions_dir, fn)

    min_index = min(dict['events'].shape[0], dict['doas'].shape[0])

    with open(file_path, 'w') as f:
        for n in range(min_index):
            event_indexes = np.where(dict['events'][n]==1.0)[0]
            azi = np.around(dict['doas'][n, event_indexes] * 180 / np.pi, 
                decimals=-1)
            ele = np.around(dict['doas'][n, event_indexes+dict['events'].shape[1]] * 180 / np.pi, 
                decimals=-1)
            for idx, k in enumerate(event_indexes):
                f.write('{},{},{},{}\n'.format(n, k, int(azi[idx]), int(ele[idx])))


def get_nb_files(_pred_file_list, _group='split'):
    '''Get attributes number
    https://github.com/sharathadavanne/seld-dcase2019/blob/master/calculate_SELD_metrics.py
    '''
    _group_ind = {'split': 5, 'ir': 9, 'ov': 13}
    _cnt_dict = {}
    for _filename in _pred_file_list:

        if _group == 'all':
            _ind = 0
        else:
            _ind = int(_filename[_group_ind[_group]])

        if _ind not in _cnt_dict:
            _cnt_dict[_ind] = []
        _cnt_dict[_ind].append(_filename)

    return _cnt_dict


def calculate_SELD_metrics(gt_meta_dir, pred_meta_dir, score_type):
    '''Calculate metrics using official tool. This part of code is modified from:
    https://github.com/sharathadavanne/seld-dcase2019/blob/master/calculate_SELD_metrics.py
    
    Args:
      gt_meta_dir: ground truth meta directory. 
      pred_meta_dir: prediction meta directory.
      score_type: 'all', 'split', 'ov', 'ir'
      
    Returns:
      metrics: dict
    '''
    
    # Load feature class
    feat_cls = cls_feature_class.FeatureClass()

    # collect gt files info
    # gt_meta_files = [fn for fn in os.listdir(gt_meta_dir) if fn.endswith('.csv') and not fn.startswith('.')]

    # collect pred files info
    pred_meta_files = [fn for fn in os.listdir(pred_meta_dir) if fn.endswith('.csv') and not fn.startswith('.')]

    # Load evaluation metric class
    eval = evaluation_metrics.SELDMetrics(
        nb_frames_1s=feat_cls.nb_frames_1s(), data_gen=feat_cls)
    
    # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)
    # score_type = 'all', 'split', 'ov', 'ir'
    split_cnt_dict = get_nb_files(pred_meta_files, _group=score_type)

    sed_error_rate = []
    sed_f1_score = []
    doa_error = []
    doa_frame_recall = []
    seld_metric = []

    # Calculate scores across files for a given score_type
    for split_key in np.sort(list(split_cnt_dict)):
        eval.reset()    # Reset the evaluation metric parameters
        for _, pred_file in enumerate(split_cnt_dict[split_key]):
            # Load predicted output format file
            pred_dict = evaluation_metrics.load_output_format_file(os.path.join(pred_meta_dir, pred_file))

            # Load reference description file
            gt_desc_file_dict = feat_cls.read_desc_file(os.path.join(gt_meta_dir, pred_file.replace('.npy', '.csv')))

            # Generate classification labels for SELD
            gt_labels = feat_cls.get_clas_labels_for_file(gt_desc_file_dict)
            pred_labels = evaluation_metrics.output_format_dict_to_classification_labels(pred_dict, feat_cls)

            # Calculated SED and DOA scores
            eval.update_sed_scores(pred_labels.max(2), gt_labels.max(2))
            eval.update_doa_scores(pred_labels, gt_labels)

        # Overall SED and DOA scores
        sed_er, sed_f1 = eval.compute_sed_scores()
        doa_err, doa_fr = eval.compute_doa_scores()
        seld_metr = evaluation_metrics.compute_seld_metric(
            [sed_er, sed_f1], [doa_err, doa_fr])

        sed_error_rate.append(sed_er)
        sed_f1_score.append(sed_f1)
        doa_error.append(doa_err)
        doa_frame_recall.append(doa_fr)
        seld_metric.append(seld_metr)

    sed_scores = [sed_error_rate, sed_f1_score]
    doa_er_metric = [doa_error, doa_frame_recall]

    sed_scores = np.array(sed_scores).squeeze()
    doa_er_metric = np.array(doa_er_metric).squeeze()
    seld_metric = np.array(seld_metric).squeeze()

    return sed_scores, doa_er_metric, seld_metric


def calculate_SELD_metrics20(sed_pred, sed_gt, doa_pred, doa_gt, param20, cur_epoch):
    '''Calculate metrics using official tool. This part of code is modified from:
    https://github.com/sharathadavanne/seld-dcase2019/blob/master/calculate_SELD_metrics.py

    Args:
      gt_meta_dir: ground truth meta directory.
      pred_meta_dir: prediction meta directory.
      score_type: 'all', 'split', 'ov', 'ir'

    Returns:
      metrics: dict
    '''
    best_seld_metric = param20['best_seld_metric']
    best_epoch = param20['best_epoch']
    patience_cnt = param20['patience_cnt']
    new_metric = param20['new_metric']
    new_seld_metric = param20['new_seld_metric']

    cls_new_metric = SELD_evaluation_metrics.SELDMetrics_custom(nb_classes=14,
                                                                doa_threshold=20,
                                                                frames_per_1s=param20['frames_per_1s'])
    pred_dict = cls_new_metric.regression_label_format_to_output_format(
        sed_pred, doa_pred
    )
    gt_dict = cls_new_metric.regression_label_format_to_output_format(
        sed_gt, doa_gt
    )

    pred_blocks_dict = cls_new_metric.segment_labels(pred_dict, sed_pred.shape[0])
    gt_blocks_dict = cls_new_metric.segment_labels(gt_dict, sed_gt.shape[0])

    cls_new_metric.update_seld_scores(pred_blocks_dict, gt_blocks_dict)
    # new_metric[cur_epoch, :] = cls_new_metric.compute_seld_scores()
    new_metric = cls_new_metric.compute_seld_scores()
    # new_seld_metric[cur_epoch] = cls_new_metric.early_stopping_metric(new_metric[cur_epoch, :2], new_metric[cur_epoch, 2:])
    new_seld_metric = cls_new_metric.early_stopping_metric(new_metric[:2], new_metric[2:])

    # Visualize the metrics with respect to epochs #
    #plot_functions(unique_name, tr_loss, sed_metric, doa_metric, seld_metric, new_metric, new_seld_metric)

    #patience_cnt += 1
    # if new_seld_metric[cur_epoch] < best_seld_metric:
    #     best_seld_metric = new_seld_metric[cur_epoch]
    #     best_epoch = cur_epoch
    #     model.save(model_name)
    #     patience_cnt = 0

    #if patience_cnt > params['patience']:
    #    break

    ############

    # print(
    #     'epoch_cnt: {}, time: {:0.2f}s, tr_loss: {:0.2f}, '
    #     '\n\t\t DCASE2019 SCORES: ER: {:0.2f}, F: {:0.1f}, DE: {:0.1f}, FR:{:0.1f}, seld_score: {:0.2f}, '
    #     '\n\t\t DCASE2020 SCORES: ER: {:0.2f}, F: {:0.1f}, DE: {:0.1f}, DE_F:{:0.1f}, seld_score (early stopping score): {:0.2f}, '
    #     'best_seld_score: {:0.2f}, best_epoch : {}\n'.format(
    #         cur_epoch, time.time() - start, tr_loss[cur_epoch],
    #         sed_metric[cur_epoch, 0], sed_metric[cur_epoch, 1] * 100,
    #         doa_metric[cur_epoch, 0], doa_metric[cur_epoch, 1] * 100, seld_metric[cur_epoch],
    #         new_metric[cur_epoch, 0], new_metric[cur_epoch, 1] * 100,
    #         new_metric[cur_epoch, 2], new_metric[cur_epoch, 3] * 100,
    #         new_seld_metric[cur_epoch], best_seld_metric, best_epoch
    #     )
    # )
    #
    # avg_scores_val.append([new_metric[best_epoch, 0], new_metric[best_epoch, 1], new_metric[best_epoch, 2],
    #                        new_metric[best_epoch, 3], best_seld_metric])
    # print('\nResults on validation split:')
    # print('\tUnique_name: {} '.format(unique_name))
    # print('\tSaved model for the best_epoch: {}'.format(best_epoch))
    # print('\tSELD_score (early stopping score) : {}'.format(best_seld_metric))
    #
    # print('\n\tDCASE2020 scores')
    # print('\tClass-aware localization scores: DOA_error: {:0.1f}, F-score: {:0.1f}'.format(new_metric[best_epoch, 2],
    #                                                                                        new_metric[best_epoch, 3] * 100))
    # print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(new_metric[best_epoch, 0],
    #                                                                                         new_metric[
    #                                                                                             best_epoch, 1] * 100))

    # print('\n\tDCASE2019 scores')
    # print('\tLocalization-only scores: DOA_error: {:0.1f}, Frame recall: {:0.1f}'.format(doa_metric[best_epoch, 0],
    #                                                                                 doa_metric[best_epoch, 1] * 100))
    # print('\tDetection-only scores: Error rate: {:0.2f}, F-score: {:0.1f}\n'.format(sed_metric[best_epoch, 0],
    #                                                                                 sed_metric[best_epoch, 1] * 100))

    return new_metric, new_seld_metric