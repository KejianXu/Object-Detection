import os

import cv2

imgsPath = r'../data/test'
outputPath = '../data/test_output'
imgs = os.listdir(imgsPath)
for img in imgs:
    imgPath = os.path.join(imgsPath, img)
    outImgPath = os.path.join(outputPath, "2560"+img)
    image = cv2.imread(imgPath)
    if image is None:
        continue
    h = image.shape[0]
    w = image.shape[1]
    pad_h = max(int((1440 - image.shape[0]) / 2),0)
    pad_w = max(int((2560 - image.shape[1]) / 2),0)
    padding_image = cv2.copyMakeBorder(src=image, top=pad_h,bottom=pad_h,left=pad_w,right=pad_w,borderType=cv2.BORDER_REPLICATE)
    cv2.imwrite(outImgPath, padding_image)