import argparse
import cv2
import glob
import os
import sys
from pathlib import Path

import scipy.io as scio
import numpy as np
import random
import time

import torch
from torch._C import StringType

from framework import Stylization




## -------------------
##  Parameters

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content video, dir of frame images or image')
parser.add_argument('--style', type=str,
                    help='File path to the style video or single image')
parser.add_argument('--model', type=str, default='Model/style_net-TIP-final.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--content_scale', type=float, default=0,
                    help='Scale the size of the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_scale', type=float, default=0,
                    help='Scale the size of the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--fps', type=int, default=30,
                    help='Frames per second of video to save')
# parser.add_argument('--save_ext', default='.mp4',
#                     help='The extension name of the output video')
parser.add_argument('--output', type=str, default='',
                    help='Directory to save the output image(s)')
parser.add_argument('--frames', type=int, default=0,
                    help='Number of frames to render, \
                    entire video if set to 0')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable cuda')
parser.add_argument('--no_global', action='store_true',
                    help='Disable global feature sharing')

args = parser.parse_args()
print(args)

# # Target style
# if len(sys.argv) == 3:
#     style_img = sys.argv[1]
#     content_video = sys.argv[2]
# else:

#     style_img = './inputs/input.jpg'

#     # Target content video
#     # Use glob.glob() to search for all the frames
#     # Add sort them by sort()
#     content_video = './inputs/content/*.png'

# # Path of the checkpoint (please download and replace the empty file)
# #checkpoint_path = "./Model/style_net-TIP-final.pth"
# #checkpoint_path = "./Model/style_net-trees-13.pth"
# checkpoint_path = "./Model/style_net-nebula-2.pth"

content_video = args.content
style_img = args.style
checkpoint_path = args.model

print(f"Style: {style_img}\nContent: {content_video}\nCheckpoint: {checkpoint_path}")

# Device settings, use cuda if available
cuda = torch.cuda.is_available() and not args.no_cuda


# The proposed Sequence-Level Global Feature Sharing
use_Global = not args.no_global

# to reduce memory limit the number of global samples
max_global_samples = 4 # None

# Saving settings
save_video = True
fps = args.fps

# Where to save the results
result_frames_path = './result_frames/'
result_videos_path = os.path.join('./result_videos/', args.output)

# scaling (use 0 to keep original scales)
style_scale = args.style_scale
content_scale = args.content_scale
# TODO: handle size

## -------------------
##  Tools


if not os.path.exists(result_frames_path):
    os.mkdir(result_frames_path)

if not os.path.exists(result_videos_path):
    os.mkdir(result_videos_path)


def read_img(img_or_path, scale=None):
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_UNCHANGED)
    else:
        img = img_or_path

    if scale is not None and scale > 0 and scale != 1:
        if scale < 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    return img


class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape

        if self.record_H == 0 and self.record_W == 0:
            new_H = H + 128
            if new_H % 64 != 0:
                new_H += 64 - new_H % 64

            new_W = W + 128
            if new_W % 64 != 0:
                new_W += 64 - new_W % 64

            self.record_H = new_H
            self.record_W = new_W

        new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H,
                                          64, self.record_W-64-W, cv2.BORDER_REFLECT)
        return new_img




## -------------------
##  Preparation


# Read style image
if not os.path.exists(style_img):
    exit('Style image %s not exists'%(style_img))

style_frames = []
if style_img.endswith(".mp4"):
    vid = cv2.VideoCapture(style_img)
    while(True):
        ret, content_img = vid.read()
        if not ret:
            break
        # pre-scale image to save memory
        scaled_img = read_img(cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB), style_scale)
        style_frames.append(scaled_img)
    style = style_frames[0]
else:
    style = read_img(style_img, style_scale)

# Build model
framework = Stylization(checkpoint_path, cuda, use_Global)
framework.prepare_style(style)

# Read content frames
if content_video.endswith(".mp4"):
    vid = cv2.VideoCapture(content_video)
    frame_list = []
    while(True):
        ret, content_img = vid.read()
        if not ret:
            break
        # pre-scale image to save memory
        scaled_img = read_img(cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB), content_scale)
        frame_list.append(scaled_img)
    content_scale = 1.0 # pre-scaled

elif "*" in content_video:
    frame_list = glob.glob(content_video)
    frame_list.sort()
elif content_video.endswith('jpg') or content_video.endswith("png"):
    frame_list = [content_video]
    save_video = False
else:
    print("Invlaid --content. Image file, file glob pattern or .mp4 video")
    sys.exit()

if len(frame_list) == 0:
    print("No frames found, content path should be a glob: `path/to/*.jpg` (and use quotes in cli)")
    sys.exit()

style_length = len(style_frames)
if style_length > 0 and (args.frames > 0 and args.frames > style_length) and style_length < len(frame_list):
    print("Style video must be the same length or longer than content video")
    sys.exit()

# Name for this testing
style_name = Path(style_img).stem
video_name = Path(content_video).stem
name = 'ReReVST-' + style_name + '-' + video_name
if not use_Global:
    name += '_no-global'
if style_scale != 0:
    name += '_style={:.1f}'.format(style_scale)
if content_scale != 0:
    name += '_content={:.1f}'.format(content_scale)

# Mkdir corresponding folders
if not os.path.exists('{}/{}'.format(result_frames_path,name)):
    os.mkdir('{}/{}'.format(result_frames_path,name))

# Build tools
reshape = ReshapeTool()



## -------------------
##  Inference

frame_num = args.frames
if frame_num <= 0:
    frame_num = len(frame_list)

# Prepare for proposed Sequence-Level Global Feature Sharing

if use_Global and len(frame_list) > 1:

    print('Preparations for Sequence-Level Global Feature Sharing')
    framework.clean()
    interval = 8
    if max_global_samples is not None:
        interval = max(8, frame_num // max_global_samples) # cap to prevent OOM errors
    sample_sum = (frame_num-1)//interval
    
    for s in range(sample_sum):
        i = s * interval
        print('Add frame %d , %d frames in total'%(s, sample_sum))
        input_frame = read_img(frame_list[i], content_scale)
        # scale down to fit into memory
        #if input_frame.shape[0] >= 1080 or input_frame.shape[1] >= 1920:
        #    input_frame = read_img(input_frame, 0.5)
        framework.add(input_frame)

    input_frame = read_img(frame_list[-1], content_scale)
    # scale down to fit into memory
    #if input_frame.shape[0] >= 1080 or input_frame.shape[1] >= 1920:
    #    input_frame = read_img(input_frame, 0.5)
    framework.add(input_frame)

    print('Computing global features')
    framework.compute()

    print('Preparations finish!')



# Main stylization

for i in range(frame_num):

    print("Stylizing frame %d"%(i))

    # Read the image
    input_frame = read_img(frame_list[i], content_scale)

    # Crop the image
    H,W,C = input_frame.shape
    new_input_frame = reshape.process(input_frame)

    # Stylization
    if len(style_frames) > 0:
        framework.prepare_style(style_frames[i])

    styled_input_frame = framework.transfer(new_input_frame)

    # Crop the image back
    styled_input_frame = styled_input_frame[64:64+H,64:64+W,:]

    # Save result
    cv2.imwrite('{}/{}/{:04d}.png'.format(result_frames_path, name, i), 
        styled_input_frame)


# Write images back to video

if save_video:
    frame_list = glob.glob("{}/{}/*.*".format(result_frames_path,name))
    frame_list.sort()
    demo = cv2.imread(frame_list[0])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter('{}/{}.mp4'.format(result_videos_path, name), 
                                   fourcc, fps, (demo.shape[1],demo.shape[0]))

    for frame in frame_list:
        videoWriter.write(cv2.imread(frame))
    videoWriter.release()
