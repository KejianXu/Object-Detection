# -*- coding: utf-8 -*-
# 
# # YOLOv5 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2, datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
sys.path.append('/usr/sbin')
from models.common import DetectMultiBackend
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync


class Yolov5Model():
    @torch.no_grad()
    def __init__(self, 
            weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    def inference(self, im0):

        # Padded resize
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]

        # Convert
        im = im.transpose((2, 0, 1))  # HWC to CHW, pillow no need BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)

        outputs = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    line = (int(cls), list(map(int, torch.stack(xyxy).cpu().numpy().tolist())), float(conf), self.names[int(cls)])  # label format
                    outputs.append(line)

        return outputs


# model = None

# def check_requirements():
#     global model
#     if model == None:
#         if os.path.exists("/usr/sbin/license.so"):
#             from license import check_license
#             if check_license():
#                 model = Yolov5Model(weights='runs/train/exp9/weights/best.pt', data='data/gy.yaml', imgsz=[640, 640], device='0', conf_thres=0.1)  
#     return True

# def predict(img):
#     global model
#     outputs = [None]
#     if model != None:
#         outputs = model.inference(img)
#     return outputs
 

'''
    predictor = YOLOXPredictor()

    outputs = predictor.inference(image_name)
'''

if __name__ == "__main__":

    from PIL import Image, ImageDraw, ImageFont
    import time
    model = Yolov5Model(weights='yolov5s.pt', data='data/coco.yaml', imgsz=[640, 640], device='0', conf_thres=0.25)

    img = "dog.jpg"
    img = cv2.imread(img)
    # img = Image.open(img)
    for i in range(10):
        t0 = time.time()
        outputs = model.inference(img)
        print(f'{int((time.time() - t0) * 1000)} ms')

    print(outputs)

    '''
    GeForce RTX 2080 Ti, 11264MiB
    23 ms
14 ms
15 ms
16 ms
15 ms
14 ms
14 ms
17 ms
14 ms
16 ms

'''