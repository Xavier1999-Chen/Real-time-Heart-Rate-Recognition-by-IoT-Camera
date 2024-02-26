"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.

"""
import csv
import glob
import os
import re
from math import ceil
from scipy import signal
from scipy import sparse
# from unsupervised_methods.methods import POS_WANG
# from unsupervised_methods import utils
import math
# from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm



def nn_preprocess(frames, W,H,Data_Types,chunk_len):
    """Preprocesses a pair of data.

    Args:
        frames(np.array): Frames in a video.
        bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
        config_preprocess(CfgNode): preprocessing settings(ref:config.py).
    Returns:
        frame_clips(np.array): processed video data by frames
        bvps_clips(np.array): processed bvp (ppg) labels by frames
    """
    # resize frames and crop for face region
    frames = crop_face_resize(
        frames,use_face_detection=True,use_larger_box=True,larger_box_coef=1.5,use_dynamic_detection=False,
        detection_freq=30,use_median_box=False, width=W, height=H)
    
    # Check data transformation type
    data = list()  # Video data
    for data_type in Data_Types:
        f_c = frames.copy()
        if data_type == "Raw":
            data.append(f_c)
        elif data_type == "DiffNormalized":
            data.append(diff_normalize_data(f_c))
        elif data_type == "Standardized":
            data.append(standardized_data(f_c))
        else:
            raise ValueError("Unsupported data type!")
        
    data = np.concatenate(data, axis=-1)  # concatenate all channels
    # if config_preprocess.LABEL_TYPE == "Raw":
    #     pass
    # elif config_preprocess.LABEL_TYPE == "DiffNormalized":
    #     bvps = BaseLoader.diff_normalize_label(bvps)
    # elif config_preprocess.LABEL_TYPE == "Standardized":
    #     bvps = BaseLoader.standardized_label(bvps)
    # else:
    #     raise ValueError("Unsupported label type!")

    # if config_preprocess.DO_CHUNK:  # chunk data into snippets
    frames_clips = chunk(data,chunk_len)


    return frames_clips

def face_detection(frame, use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.

    Args:
        frame(np.array): a single frame.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """

    detector = cv2.CascadeClassifier(
        'neural_methods/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if len(face_zone) < 1:
        print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        face_box_coor = np.argmax(face_zone, axis=0)
        face_box_coor = face_zone[face_box_coor[2]]
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    else:
        face_box_coor = face_zone[0]
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor

def crop_face_resize(frames, use_face_detection, use_larger_box, larger_box_coef, use_dynamic_detection, 
                        detection_freq, use_median_box, width, height):
    """Crop face and resize frames.

    Args:
        frames(np.array): Video frames.
        use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                        and resizing.
                                        If True, it performs face detection every "detection_freq" frames.
        detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
        width(int): Target width for resizing.
        height(int): Target height for resizing.
        use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
        use_face_detection(bool):  Whether crop the face.
        larger_box_coef(float): the coefficient of the larger region(height and weight),
                            the middle point of the detected region will stay still during the process of enlarging.
    Returns:
        resized_frames(list[np.array(float)]): Resized and cropped frames
    """
    # Face Cropping
    if use_dynamic_detection:
        num_dynamic_det = ceil(frames.shape[0] / detection_freq)
    else:
        num_dynamic_det = 1
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    for idx in range(num_dynamic_det):
        if use_face_detection:
            face_region_all.append(face_detection(frames[detection_freq * idx], use_larger_box, larger_box_coef))
        else:
            face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region_all, dtype='int')
    if use_median_box:
        # Generate a median bounding box based on all detected face regions
        face_region_median = np.median(face_region_all, axis=0).astype('int')


    # Frame Resizing
    resized_frames = np.zeros((frames.shape[0], height, width, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
            reference_index = i // detection_freq
        else:  # use the first region obtrained from the first frame.
            reference_index = 0
        if use_face_detection:
            if use_median_box:
                face_region = face_region_median
            else:
                face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frames

def chunk(frames, chunk_length):
    """Chunk the data into small chunks.

    Args:
        frames(np.array): video frames.
        bvps(np.array): blood volumne pulse (PPG) labels.
        chunk_length(int): the length of each chunk.
    Returns:
        frames_clips: all chunks of face cropped frames
        bvp_clips: all chunks of bvp frames
    """

    clip_num = frames.shape[0] // chunk_length
    frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    return np.array(frames_clips)


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


def standardized_data(data):
    """Z-score standardization for video data."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data

