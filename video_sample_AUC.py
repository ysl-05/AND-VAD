"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

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


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='/')
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

    # loss
    score_func = nn.MSELoss(reduction="none")  # KEEP LOSS UNIFY
    #dataload
    test_dataset = frames_DataLoader(args.test_frames, image_size=64, time_step=args.seq_len)

    test_batch = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('len(test_batch)', len(test_batch))
    #--------------MNAD AUC CODE------------------
    labels = np.load(args.label_file)
    print('labels', len(labels))
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
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]

        labels_list = np.append(labels_list, labels[4 + label_length:videos[video_name]['length'] + label_length -1])

        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    # ----------------testing ---------------
    all_videos = []
    all_gt = []
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    
    for j, (data) in enumerate(test_batch):
        if args.cond_generation:
            # video, _ = next(data)
            video = data.permute(0, 4, 1, 2,3)
            print('data shape', video.shape)
            cond_kwargs["cond_img"] = video[:,:,cond_frames].to(dist_util.dev())

            pred_gt= video[:, :, ref_frames].to(dist_util.dev())
            video = video.to(dist_util.dev())
            print('cond_kwargs["cond_img"]', cond_kwargs["cond_img"].shape, pred_gt.shape)

        logger.log("sampling...")
        sample = sample_fn(
            model,
            (args.batch_size, channels, args.seq_len, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            progress=False,
            cond_kwargs=cond_kwargs
        )
        # 
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        pred_out = sample[:, :, ref_frames]
        # sample = sample.permute(0, 2, 3, 4, 1)
        print('sample result shape', sample.shape)
        pred_out_image = pred_out.permute(0, 2, 3, 4, 1)
        print('predict result shape', pred_out_image.shape)

        logger.log("sampling complete")

        if args.visualize:
            pred_out = pred_out.permute(0, 2, 3, 4, 1).contiguous()
            gathered_samples = [th.zeros_like(pred_out) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, pred_out)  # gather not supported with NCCL
            all_videos.extend([pred_out.cpu().numpy() for pred_out in gathered_samples])
            # logger.log(f"created {len(all_videos) * args.batch_size} samples")
            # arr = np.concatenate(all_videos, axis=0)
        #--------------- loss calculate------------------
        # step 1
        loss_frame_test = score_func(pred_out, pred_gt)  # MSE  Better
        # loss_frame_test = torch.abs(out['prediction'] - x['driving'])  # ABS Worse

        # step 2
        frame_scores = th.mean(loss_frame_test).item()
        # frame_scores = torch.sum(torch.sum(torch.sum(loss_frame_test, axis=3), axis=2), axis=1).item()

        # if training_stats_path is not None:
        #     # mean-std normalization
        #     # print('load the training score ...')
        #     scores = (frame_scores - frame_mean) / frame_std
        # else:
        #     scores = (frame_scores - 0.9246) / 1.9988
        #     # print('frame score', scores, len(frame_scores), frame_scores)
        #
        # for i in range(len(scores)):
        #     frame_bbox_scores[j][i] = scores[i]

        # ------------------------------------
        # if args.method == 'pred':

        if j == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(frame_scores))

       

        if args.cond_generation and args.save_gt:
            video = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8)
            video = video.permute(0, 2, 3, 4, 1)
            video = video.contiguous()

            # gathered_videos = [th.zeros_like(video) for _ in range(dist.get_world_size())]
            gathered_videos = th.zeros_like(video)
            # dist.all_gather(gathered_videos, video)  # gather not supported with NCCL
            all_gt.extend([video.cpu().numpy() for video in gathered_videos])
            logger.log(f"created {len(all_gt) * args.batch_size} videos")

    del test_dataset



    # visualization
    # A = np.load('')
    if args.visualize:
        arr = np.concatenate(all_videos, axis=0)
        save_path = '' #
        # data = A['arr_0']
        last_folder = 'test_' + args.model_path.split('/')[-1].split('.')[0]
        save_visual_path = os.path.join(save_path, last_folder)
        if not os.path.exists(save_visual_path):
            os.makedirs(save_visual_path)
            print(save_visual_path)
        print('array shape is:', arr.shape)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                im = Image.fromarray(arr[i][j])
                # im.show()
                image_name = str(i) + str(j) + '.jpg'
                im.save(save_visual_path+ '/' + image_name)
    if args.cond_generation and args.save_gt:
        arr_gt = np.concatenate(all_gt, axis=0)

    # if dist.get_rank() == 0 and args.save_gt:
    #
    #     # shape_str = "x".join([str(x) for x in arr.shape])
    #     # logger.log(f"saving samples to {os.path.join(logger.get_dir(), shape_str)}")
    #     # np.savez(os.path.join(logger.get_dir(), shape_str), arr)
    #
    #     if args.cond_generation and args.save_gt:
    #         shape_str_gt = "x".join([str(x) for x in arr_gt.shape])
    #         logger.log(f"saving ground_truth to {os.path.join(logger.get_dir(), shape_str_gt)}")
    #         np.savez(os.path.join(logger.get_dir(), shape_str_gt), arr_gt)

    # dist.barrier()
    #------------------------
    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []

    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        # print('video_name', video_name)
        subvideo_list = psnr_list[video_name]
        for i in range(len(psnr_list[video_name])):
            anomaly_score_total_list.append(anomaly_score(subvideo_list[i], np.max(subvideo_list), np.min(subvideo_list)))
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    # labels_list = labels_list[:-6]
    # print(anomaly_score_total_list.shape, labels_list.shape)
    accuracy = AUC(anomaly_score_total_list[:len(test_batch)], np.expand_dims(1 - labels_list, 0)[:len(test_batch)])

    print('AUC score by MNAD method: ', accuracy * 100, '%')

def create_argparser():
    defaults = dict(
        visualize=True,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path='', 
        seq_len=4,
        sampling_type="generation",
        cond_frames="0,1,2,3",
        cond_generation=True,
        resample_steps=1, 
        data_dir=''
        label_file='',
        save_gt=False,
        seed=0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
