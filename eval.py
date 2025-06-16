import argparse
import os
import torch
import cv2
import joblib
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from models.mem_cvae import HFVAD
from datasets.dataset import Chunked_sample_dataset
from utils.eval_utils import save_evaluation_curves
from pylab import *
from torchstat import stat
from thop import profile
# from thop import profile

METADATA = {    
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    }
}

def visualize_feature_map(img_batch, target_batch, num_id):
    feature_map = np.squeeze(img_batch)
    frame_traget = np.squeeze(target_batch)
    plt.figure()
    print('feature_map', feature_map.shape, frame_traget.shape)
    feature_map = np.squeeze(img_batch)
    frame_target = np.squeeze(target_batch)
    plt.figure()
    # print('feature_map', feature_map.shape, frame_traget.shape)

    feature_map_split = feature_map[1, :, :]  # (3, 32, 32)
    target_map_split = frame_target[1, :, :]  # (3, 32, 32)
    feature_map_error_split = feature_map_split - target_map_split
    # (3, 32, 32)
    feature_map = feature_map.transpose((1, 2, 0))
    frame_target = frame_target.transpose((1, 2, 0))
    feature_map_error = feature_map - frame_target

    # b, g, r = cv2.split(feature_map_split.transpose((1, 2, 0)) )  # 分别提取B、G、R通道
    # feature_map_split = cv2.merge([r, g, b])   # 重新组合为R、G、B

    # feature_map = feature_map[:, :, ::-1]
    # # plt.xticks([]), plt.yticks([])  # 隐藏x和y轴
    # plt.imshow(feature_map, cmap = plt.cm.jet)

    # plt.imshow(feature_map_error[:,:,::-1])
    # plt.imshow(feature_map_error_split)
    plt.savefig(".data/avenue/feature_map_avenue/" + str(num_id) + "_feature_map.jpg")
    plt.show()

def evaluate(config, ckpt_path, testing_chunked_samples_file, training_stats_path, suffix):
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    device = config["device"]
    num_workers = config["num_workers"]

    testset_num_frames = np.sum(METADATA[dataset_name]["testing_frames_cnt"])

    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
                  num_pred=config["model_paras"]["clip_pred"],
                  config=config,
                  features_root=config["model_paras"]["feature_root"],
                  num_slots=config["model_paras"]["num_slots"],
                  shrink_thres=config["model_paras"]["shrink_thres"],
                  mem_usage=config["model_paras"]["mem_usage"],
                  skip_ops=config["model_paras"]["skip_ops"],
                  ).to(device).eval()

    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    # print("load pre-trained success!")

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))


    #  get training stats
    if training_stats_path is not None:
        training_scores_stats = torch.load(training_stats_path)

        of_mean, of_std = np.mean(training_scores_stats["of_training_stats"]), \
                          np.std(training_scores_stats["of_training_stats"])
        frame_mean, frame_std = np.mean(training_scores_stats["frame_training_stats"]), \
                                np.std(training_scores_stats["frame_training_stats"])

    score_func = nn.MSELoss(reduction="none")

    # 文件读取
    frame_bbox_scores = [{} for i in range(testset_num_frames.item())]

    num_id = 1
    for idx, testing_chunked_samples_file in enumerate(testing_chunked_samples_dir):
        testing_chunked_samples_file = os.path.join("./data", config["dataset_name"],
                                                    "testing/chunked_samples_32/", testing_chunked_samples_file)
        dataset_test = Chunked_sample_dataset(testing_chunked_samples_file)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, num_workers=num_workers, shuffle=False)
        # bbox anomaly scores for each frame

        for test_data in tqdm(dataloader_test, desc="Eval: ", total=len(dataloader_test)):

            sample_frames_test, sample_ofs_test, bbox_test, pred_frame_test, indices_test = test_data
            sample_frames_test = sample_frames_test.to(device)
            sample_ofs_test = sample_ofs_test.to(device)
            # print('sample_frames_test', sample_frames_test.shape, bbox_test, 'pred_frame_test', pred_frame_test.shape, pred_frame_test, indices_test)
           # measure the FlOPs parameter
           #  macs, params = profile(model, inputs=(sample_frames_test, sample_ofs_test))
           #  print('macs, params', macs, params)

            # stat(model, sample_frames_test, sample_ofs_test)

            out_test = model(sample_frames_test, sample_ofs_test, mode="test")

            # visualization feature map
            num_id += 1
            img_batch = np.expand_dims(out_test["frame_pred"].cpu().numpy(), axis=0)
            target_batch = np.expand_dims(out_test["frame_target"].cpu().numpy(), axis=0)
            # print("conv_img=", out_test["frame_pred"].shape)
            visualize_feature_map(img_batch, target_batch, num_id)

            # loss_of_test = score_func(out_test["of_recon"], out_test["of_target"]).cpu().data.numpy()
            loss_frame_test = score_func(out_test["frame_pred"], out_test["frame_target"]).cpu().data.numpy()

            # print('\n out_test["frame_pred"], out_test["frame_target"]', out_test["frame_pred"].shape, out_test["frame_target"].shape, 'loss_frame_test', loss_frame_test.shape)

            # of_scores = np.sum(np.sum(np.sum(loss_of_test, axis=3), axis=2), axis=1)
            frame_scores = np.sum(np.sum(np.sum(loss_frame_test, axis=3), axis=2), axis=1)
            print('frame_scores', frame_scores)
            if training_stats_path is not None:
                # mean-std normalization
                # print(of_std.shape, of_std)

                # of_scores = (of_scores - of_mean) / of_std
                frame_scores = (frame_scores - frame_mean) / frame_std
            print('frame_mean, frame_std', frame_mean, frame_std)
            scores = config["w_p"] * frame_scores  # + config["w_r"] * of_scores
            # print('len(scores)', len(scores))
            for i in range(len(scores)):
                frame_bbox_scores[pred_frame_test[i][-1].item()][i] = scores[i]


    del dataset_test

    # joblib.dump(frame_bbox_scores,
    #             os.path.join(config["eval_root"], config["exp_name"], "frame_bbox_scores_%s.json" % suffix))

    # frame_bbox_scores = joblib.load(os.path.join(config["eval_root"], config["exp_name"],
    #                                              "frame_bbox_scores_%s.json" % suffix))

    # frame-level anomaly score
    frame_scores = np.empty(len(frame_bbox_scores))
    for i in range(len(frame_scores)):
        if len(frame_bbox_scores[i].items()) == 0:
            frame_scores[i] = config["w_r"] * (0 - of_mean) / of_std + config["w_p"] * (0 - frame_mean) / frame_std
        else:
            frame_scores[i] = np.max(list(frame_bbox_scores[i].values()))

    joblib.dump(frame_scores,
                os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix))
    print("len(frame_bbox_scores)", len(frame_bbox_scores))
    # frame_scores = joblib.load(
    #     os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix)
    # )

    # ================== Calculate AUC ==============================
    # load gt labels
    gt = pickle.load(
        open(os.path.join(config["dataset_base_dir"], "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
    gt_concat = np.concatenate(list(gt.values()), axis=0)
    print("len(gt_concat)", len(gt_concat))
    new_gt = np.array([])
    new_frame_scores = np.array([])

    start_idx = 0
    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
        gt_each_video = gt_concat[start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        scores_each_video = frame_scores[
                            start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]

        start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores
    print('frame score shape', frame_scores.shape)
    normal_score_sum = 0
    abnormal_score_sum = 0
    normal_frame_sum =0
    abnormal_frame_sum=0

    for id in range(len(frame_scores)):
        if gt_concat[id] == 1:
            normal_score_sum += frame_scores[id]
            normal_frame_sum += 1
        else:
            abnormal_score_sum += frame_scores[id]
            abnormal_frame_sum += 1
    print(normal_score_sum, normal_frame_sum, abnormal_score_sum, abnormal_frame_sum)
    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'anomaly_curves_%s' % suffix)
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 4)

    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str,
                        default="",
                        help='path to pretrained weights')
    parser.add_argument("--cfg_file", type=str,
                        default="./avenue.yaml",
                        help='path to pretrained model configs')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.cfg_file))

   
    testing_chunked_samples_dir = sorted(os.listdir(os.path.join("./data", config["dataset_name"],
                                                                 "testing/chunked_samples__32")))
    from train import cal_training_stats

    os.makedirs(os.path.join("./eval", config["exp_name"]), exist_ok=True)
    training_chunked_samples_dir = os.path.join("./data", config["dataset_name"],
                                                "training/chunked_samples")
    training_stat_path = os.path.join("./eval", config["exp_name"], "training_stats.npy")
    # cal_training_stats(config, args.model_save_path, training_chunked_samples_dir, training_stat_path)
    #
    with torch.no_grad():
        auc = evaluate(config, args.model_save_path,
                       testing_chunked_samples_dir,
                       training_stat_path, suffix="best")

        print(auc)


