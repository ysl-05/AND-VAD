import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import Optional, Tuple, List


def random_apply(func, p, x):
    if random.random() < p:
        return func(x)
    return x


def resize_and_rescale(x, y, img_size):
    x = TF.resize(x, [img_size, img_size])
    return x, y


def _sample_or_pad_sequence_indices(sequence: torch.Tensor, num_steps: int,
                                    stride: int,
                                    offset: torch.Tensor) -> torch.Tensor:
    sequence_length = len(sequence)
    sel_idx = torch.arange(sequence_length)

    max_length = num_steps * stride + offset
    num_repeats = torch.div(max_length + sequence_length - 1,
                                   sequence_length, rounding_mode='floor')
    sel_idx = sel_idx.repeat_interleave(num_repeats)

    steps = torch.arange(offset, offset + num_steps * stride, stride)
    return sel_idx[steps]


def sample_sequence(sequence, num_steps, random=True, stride=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    sequence_length = len(sequence)
    if random:
        max_offset = max(0, sequence_length - (num_steps - 1) * stride)
        offset = random.randint(0, max_offset)
    else:
        offset = max(0, (sequence_length - num_steps * stride) // 2)
    
    indices = _sample_or_pad_sequence_indices(
        sequence=sequence, num_steps=num_steps, stride=stride, offset=offset)
    return sequence[indices]


def random_crop_resize(frames, output_h, output_w, aspect_ratio=(0.75, 1.33), area_range=(0.5, 1)):
    seq_len, h, w, c = frames.shape
    area = h * w
    target_area = random.uniform(area_range[0], area_range[1]) * area
    aspect_ratio = random.uniform(aspect_ratio[0], aspect_ratio[1])
    
    target_w = int(round(np.sqrt(target_area * aspect_ratio)))
    target_h = int(round(np.sqrt(target_area / aspect_ratio)))
    
    if target_w > w:
        target_w = w
    if target_h > h:
        target_h = h
    
    i = random.randint(0, h - target_h)
    j = random.randint(0, w - target_w)
    
    frames = frames[:, i:i+target_h, j:j+target_w, :]
    frames = F.interpolate(frames.permute(0, 3, 1, 2), size=(output_h, output_w), mode='bilinear')
    return frames.permute(0, 2, 3, 1)


def gaussian_blur(image, kernel_size, sigma):
    channels = image.shape[-1]
    kernel = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2+1)**2 / (2*sigma**2))
    kernel = kernel / kernel.sum()
    
    kernel = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    kernel = kernel.view(channels, 1, kernel_size, 1)
    
    image = image.permute(0, 3, 1, 2)
    blurred = F.conv2d(image, kernel, padding=(kernel_size//2, 0), groups=channels)
    blurred = F.conv2d(blurred, kernel.transpose(2, 3), padding=(0, kernel_size//2), groups=channels)
    return blurred.permute(0, 2, 3, 1)


def random_blur(image, height, p=0.2):
    def _transform(image):
        sigma = random.uniform(0.1, 2.0)
        return gaussian_blur(image, kernel_size=height//20, sigma=sigma)
    return random_apply(_transform, p, image)


def random_flip_left_right(frames, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if random.random() < 0.5:
        return torch.flip(frames, [2])
    return frames


def random_flip_up_down(frames, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if random.random() < 0.5:
        return torch.flip(frames, [1])
    return frames


def random_rotation(frames, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if random.random() < 0.5:
        return torch.rot90(frames, k=1, dims=[1, 2])
    return frames


def to_grayscale(image, keep_channels=True):
    image = TF.rgb_to_grayscale(image)
    if keep_channels:
        image = image.repeat(1, 1, 1, 3)
    return image


def random_grayscale_3d(frames, p=0.2):
    num_frames, width, height, channels = frames.shape
    big_image = frames.reshape(-1, height, channels)
    big_image = random_apply(to_grayscale, p, big_image)
    return big_image.reshape(num_frames, width, height, channels)


def random_brightness(image, max_delta=0.3):
    factor = random.uniform(1.0 - max_delta, 1.0 + max_delta)
    return image * factor


def random_reverse(frames, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if random.random() < 0.5:
        return torch.flip(frames, [0])
    return frames


def random_perspective(frames, distortion_scale=0.5, p=0.5):
    if random.random() < p:
        height, width = frames.shape[1:3]
        startpoints = [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]
        endpoints = []
        for x, y in startpoints:
            dx = random.uniform(-width * distortion_scale, width * distortion_scale)
            dy = random.uniform(-height * distortion_scale, height * distortion_scale)
            endpoints.append([x + dx, y + dy])
        
        frames = frames.permute(0, 3, 1, 2)
        frames = TF.perspective(frames, startpoints, endpoints)
        return frames.permute(0, 2, 3, 1)
    return frames


def random_background_noise(frames, noise_level=0.1, p=0.3):
    if random.random() < p:
        noise = torch.randn_like(frames) * noise_level
        mask = torch.rand(frames.shape[0], 1, 1, 1) > 0.5
        frames = torch.where(mask, frames + noise, frames)
    return frames


# Arbitrary speed / frequency augmentation for SimPer
def arbitrary_speed_subsample(frames_speed, num_steps, random=True, img_size=224, channels=3, stride=1, seed=None):
    frames, speed = frames_speed
    if seed is not None:
        torch.manual_seed(seed)
    
    frame_len = len(frames)
    max_frame_len = int(frame_len / speed) if speed > 1 else frame_len
    
    x = torch.linspace(0, speed * (frame_len - 0.5), frame_len)
    new_frames = []
    for i in range(frame_len):
        idx = torch.searchsorted(x, i * speed)
        if idx >= frame_len:
            idx = frame_len - 1
        new_frames.append(frames[idx])
    
    sequence = torch.stack(new_frames)[:max_frame_len]
    
    if random:
        max_offset = max(0, len(sequence) - (num_steps - 1) * stride)
        offset = random.randint(0, max_offset)
    else:
        offset = max(0, (len(sequence) - num_steps * stride) // 2)
    
    indices = _sample_or_pad_sequence_indices(
        sequence=sequence, num_steps=num_steps, stride=stride, offset=offset)
    return sequence[indices]


# (batched) Arbitrary speed / frequency augmentation for SimPer
def batched_arbitrary_speed(frames, num_diff_speeds, speed_range=(0.5, 2)):
    random_speeds = torch.rand(num_diff_speeds) * (speed_range[1] - speed_range[0]) + speed_range[0]
    random_speeds = torch.sort(random_speeds)[0]
    random_speeds = torch.cat([random_speeds, random_speeds])
    
    batched_frames = frames.repeat(num_diff_speeds * 2, 1, 1, 1)
    batched_frames = torch.stack([arbitrary_speed_subsample((f, s), num_steps=len(frames)) 
                                for f, s in zip(batched_frames, random_speeds)])
    
    return batched_frames, random_speeds
