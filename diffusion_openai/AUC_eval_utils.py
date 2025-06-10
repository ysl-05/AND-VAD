import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.signal as signal

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(mse):
    return 10 * math.log10(1 / mse)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):
    img_re = copy.copy(img)

    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))

    return img_re


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
    normal = (1 - torch.exp(-error))
    score = (torch.sum(normal * loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))

    return list_result


def draw_roc_curve(fpr, tpr, auc, psnr_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(psnr_dir, "0_auc.png"))
    plt.close()


def nonzero_intervals(vec):
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec) == 0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    tmp1 = (vec == 0) * 1
    tmp = np.diff(tmp1)
    edges, = np.nonzero(tmp)
    edge_vec = [edges + 1]

    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])


def save_evaluation_curves(scores, labels, curves_save_path, video_frame_nums):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)

    scores = scores.flatten()
    labels = labels.flatten()

    scores_each_video = {}
    scores_each_video0 = {}
    scores_each_video1 = {}

    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        # print('label', start_idx + video_frame_nums[video_id])
        scores_each_video[video_id] = scores[start_idx:start_idx + video_frame_nums[video_id]]
        scores_each_video[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=27)
        scores_each_video0[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=17)
        scores_each_video1[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=37)
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []
    preds1 = []
    preds0 = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])
        preds1.append(scores_each_video1[i])
        preds0.append(scores_each_video0[i])
    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds0 = np.concatenate(preds0, axis=0)
    preds1 = np.concatenate(preds1, axis=0)
    # truth = labels
    # preds = scores
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=0)
    auroc = auc(fpr, tpr)
    print('hfvad auc:  ', auroc)


    fpr0, tpr0, roc_thresholds0 = roc_curve(truth, preds0, pos_label=0)
    auroc0 = auc(fpr0, tpr0)
    print('hfvad auc:  ', auroc0)
    fpr1, tpr1, roc_thresholds1 = roc_curve(truth, preds1, pos_label=0)
    auroc1 = auc(fpr1, tpr1)
    print('hfvad auc:  ', auroc1)

    # draw ROC figure
    draw_roc_curve(fpr, tpr, auroc, curves_save_path)
    for i in sorted(scores_each_video.keys()):
        plt.figure()

        x = range(0, len(scores_each_video[i]))
        # print('current i ', i)
        plt.xlim([x[0], x[-1] + 4])

        # anomaly scores
        plt.plot(x, scores_each_video[i], color="blue", lw=2, label="Anomaly Score")

        # abnormal sections
        lb_one_intervals = nonzero_intervals(labels_each_video[i])
        for idx, (start, end) in enumerate(lb_one_intervals):
            plt.axvspan(start, end, alpha=0.5, color='purple',
                        label="_" * idx + "Anomaly Intervals")

        plt.xlabel('Frames Sequence')
        plt.title('Test video #%d' % (i + 1))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(curves_save_path, "0_anomaly_curve_%d.png" % (i + 1)))
        plt.close()

    # return auroc
