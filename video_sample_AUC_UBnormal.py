
import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
import skimage
import os, sys
sys.path.insert(1, os.getcwd()) 
import random
from PIL import Image
import glob
import tqdm

from collections import OrderedDict
from diffusion_openai.AUC_eval_utils import *
from diffusion_openai.mnad_frames_data_utils import frames_DataLoader
from diffusion_openai.video_datasets import load_data
from diffusion_openai import dist_util, logger
from diffusion_openai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from object_detect.extract_bboxes import object_bbox_image
from object_detect.mmdet.apis import init_detector

def main():
    config_file = './object_detect/mmdet/configs/rtmdet/rtmdet_l_8xb32_300e_coco.py'
    checkpoint_file = './object_detect/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'

    # mm_det_model = init_detector(config_file, checkpoint_file, device="cuda:0")
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='your_path')
    if args.seed:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    cond_kwargs = {}
    cond_frames = []
    if args.cond_generation:
        num = ""
        for i in args.cond_frames:
            if i == ",":
                cond_frames.append(int(num))
                num = ""
            else:
                num = num + i
        ref_frames = list(i for i in range(args.seq_len) if i not in cond_frames)
        logger.log(f"cond_frames: {cond_frames}")
        logger.log(f"ref_frames: {ref_frames}")
        logger.log(f"seq_len: {args.seq_len}")
        cond_kwargs["resampling_steps"] = args.resample_steps
    cond_kwargs["cond_frames"] = cond_frames
    print('condition frame id', cond_frames, 'ref frame id', ref_frames)
    if args.rgb:
        channels = 3
    else:
        channels = 1
    print(args.model_path)
    # loss
    score_func = nn.MSELoss(reduction="none")  # KEEP LOSS UNIFY
    #dataload
    test_dataset = frames_DataLoader(args.test_frames, image_size=args.imagesize, time_step=args.seq_len)
    # test_dataset_group = (video_name, test_dataset)
    test_batch = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('len(test_batch)', len(test_batch))
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(args.test_frames, '*')))
    test_video_select = len(videos_list)
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []  
    label_length = args.start_frame_num  # 265 + 433 + 337 + 601
    psnr_list = {}
    mse_list = {}
    ssim_list = {}
    feature_distance_list = {}
    label_dict = {}
    # Setting for video anomaly detection
    print('len video list', len(videos_list), 'video name', videos_list)
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        print('videos[video_name]', video_name, videos[video_name]['length'])
        # print('label_length',label_length)
        # for p in range(videos[video_name]['length'] - (args.seq_len -1 + label_length)):
        #     labels_list.append(labels[args.seq_len -1 + label_length  + p])
        add_label = labels[args.seq_len - 1 + label_length:videos[video_name]['length'] + label_length]
        labels_list = np.append(labels_list, add_label)
        label_dict[video_name] = add_label.tolist()
        # if video_name == "01_0025":
        #     np.savetxt('./label_01_0025.txt', str(add_label.tolist()))
        # print('gt label:', video_name, '\n', str(add_label.tolist()))

        print('len label', video_name, "start id", label_length, 'total lable length', labels_list.shape, 'added label length', len(add_label))
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        mse_list[video_name] = []
        ssim_list[video_name] = []
        feature_distance_list[video_name] = []
    print('gt label:', video_name, '\n', '\n', label_dict)
    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    # ----------------testing ---------------
    all_videos = []
    all_gt = []
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    save_path = './path/ubnormal_test_s1_128'  
    last_folder = 'test_' + args.model_path.split('/')[-1].split('.')[0]
    save_visual_path = os.path.join(save_path, last_folder)
    if not os.path.exists(save_visual_path):
        os.makedirs(save_visual_path)
        # print(save_visual_path)
    area_list = []
    for j, (data_group) in enumerate(test_batch):
        if j == 0:
            video_name_last = data_group[0]
            j_current = 0

        video_name_current = data_group[0]
        if video_name_last != video_name_current:
            j_current = 0

        # frame_name = data_group[1]
        data_crop = data_group[1]
        # print('data shape', data_crop.shape,)
        # print('current frame from video', video_name_current[0],  '/', j, '-th frame.')

        # calculate foreground area to regulization
        root_image_dir = ''
        frame_id = 10003 + j_current
       
        # init frame id
        j_current += 1
        video_name_last = video_name_current



        ssim_score = 1
        mse_scores = 0
        # cut_imagetwo = data_crop[0].permute(1, 4, 0, 2, 3)  
        cut_imagetwo = data_crop.permute(0, 4, 1, 2, 3)
        for cut in range(1):
            if args.cond_generation:
                # video, _ = next(data)
                video = cut_imagetwo[cut].unsqueeze(0)
                # print('data shape', cut_imagetwo.shape, video.shape) #  （4，64，64，3）to（1，3，4，64，64）
                cond_kwargs["cond_img"] = video[:,:,cond_frames].to(dist_util.dev())

                video = video[:, :, ref_frames].to(dist_util.dev())  # （1，3, 1，64，64）
                pred_gt_01 = ((video + 1) * 127.5).clamp(0, 255)
                pred_gt = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8)
                video = video.to(dist_util.dev())
                # print('cond_kwargs["cond_img"]', cond_kwargs["cond_img"].shape, pred_gt.shape)

            # logger.log("sampling...")
            sample = sample_fn(
                model,
                (args.batch_size, channels, args.seq_len, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                progress=False,
                cond_kwargs=cond_kwargs
            )
            # 
            sample_out = sample[:, :, ref_frames]  # （1，3，1，64，64）\
            pred_out_01 = ((sample_out + 1) * 127.5).clamp(0, 255)
            pred_out = ((sample_out + 1) * 127.5).clamp(0, 255).to(th.uint8)  # （1，3，4，64，64）
            # print('sample result shape', sample.shape)


            # logger.log("sampling complete")
            pred_out_image = pred_out.permute(0, 2, 3, 4, 1)  # （1，1，64，64，3）
            pred_gt_image = pred_gt.permute(0, 2, 3, 4, 1)
            im = Image.fromarray(pred_out_image[0][0].cpu().numpy())
            imgt = Image.fromarray(pred_gt_image[0][0].cpu().numpy())

                       
    gt_concat = np.expand_dims(labels_list, 0)
    frame_scores = anomaly_score_total_list


    save_evaluation_curves(frame_scores, gt_concat, save_visual_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 3)

def create_argparser():
    defaults = dict(
        medfilter=False,
        visualize=True,
        ssim=True,
        use_mse=False,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=True,
        model_path="",
        seq_len=4,
        sampling_type="generation",
        cond_frames="0,1,2,",
        cond_generation=True,
        resample_steps=1, 
        data_dir=''
        test_frames=''        
        label_file='',
        start_frame_num=0,
        imagesize=128,
        save_gt=False,
        seed=123,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    end = time.time()
    print(f"elapsed time: {end - start}")