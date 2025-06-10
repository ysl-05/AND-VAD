import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
# from augmentation import AllAugmentationTransform
from PIL import Image, ImageSequence
import torchvision.transforms as transform
rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


def resize_frame_obey_RaMViD(filename, resolution):
    frame = Image.open(filename)
    while min(*frame.size) >= 2 * resolution:
        frame = frame.resize(
            tuple(x // 2 for x in frame.size), resample=Image.BOX
        )
        # print('frame size ', frame.size)
    scale = resolution / min(*frame.size)
    frame = frame.resize(
        (resolution, resolution), resample=Image.BICUBIC
    )
    print('frame size rescale', resolution, frame.size)
    arr = np.array(frame.convert("RGB"))

    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    print('crop_x crop_y', crop_x, crop_y)
    arr = arr[crop_y: crop_y + resolution, crop_x: crop_x + resolution]
    arr = arr.astype(np.float32) / 127.5 - 1
    return arr

def resize_frame_sht_twopart(filename, resolution):
    frame = Image.open(filename)  #  856, 480
    arr = np.array(frame.convert("RGB"))
    arr = arr[32:, :]

    frame = Image.fromarray(np.uint8(arr))
    # frame.show()
    # print('frame crop size ', arr.shape, frame.size)

    if resolution == 64:
        frame = frame.resize(
            tuple(round(x) for x in [120,64]), resample=Image.BICUBIC
        )
    if resolution == 128:
        frame = frame.resize(
            tuple(round(x) for x in [256,128]), resample=Image.BICUBIC
        )
    # print('frame size rescale', frame.size)
    arr = np.array(frame.convert("RGB"))

    crop_y = (arr.shape[0] - resolution)
    crop_x = (arr.shape[1] - resolution)
    # print('crop_x crop_y',  arr.shape, crop_y)
    arr1 = arr[0:resolution, 0:resolution]
    arr1 = arr1.astype(np.float32) / 127.5 - 1
    arr2 = arr[0:resolution, crop_x:]
    arr2 = arr2.astype(np.float32) / 127.5 - 1
    crop_two = [arr1, arr2]

    # frame1 = Image.fromarray(np.uint8(arr1))
    # frame1.show()
    # frame2 = Image.fromarray(np.uint8(arr2))
    # frame2.show()
    return crop_two

class frames_DataLoader(data.Dataset):
    def __init__(self, video_folder, image_size, time_step, num_pred=1):
        self.dir = video_folder
        # self.transform = transform
        self.videos = OrderedDict()
        self.transform_image = transform.Compose([transform.ToTensor()])
        # self._resize_height = resize_height
        # self._resize_width = resize_width
        self.image_size = image_size
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

        # self.augment = AllAugmentationTransform(**augmentation_params)

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self): # load all frames
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step + 1):
            # for i in range(len(self.videos[video_name]['frame'])):
                # print('statistic', self.videos[video_name]['frame'][i])
                frames.append(self.videos[video_name]['frame'][i])
                # print('selected frames  in the video', str(video_name), "length is :", len(self.videos[video_name]['frame']) - self._time_step +1)
        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        # print('video_name, frame_name', video_name, frame_name)

        batch = []
        for i in range(self._time_step):
            # print('self.videos[video_name][frame][frame_name + i]', i, self.videos[video_name]['frame'][frame_name + i])
            # image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
            #                       self._resize_width)

            # image = resize_frame_sht_twopart(self.videos[video_name]['frame'][frame_name + i], self.image_size)

            image = resize_frame_obey_RaMViD(self.videos[video_name]['frame'][frame_name + i], self.image_size)
            batch.append(image)
            # batch = np.concatenate((batch,np.expand_dims(image,axis=0)))
        # print('data input',np.array(batch).shape)
            # if self.transform is not None:
            # batch.append(self.transform_image(image))

        # if self.augment is not None:
        #     batch = self.augment(batch)

        return (video_name,  np.array(batch)) #np.concatenate(batch, axis=0)

    def __len__(self):
        print('len(self.samples)', len(self.samples))
        return len(self.samples)
