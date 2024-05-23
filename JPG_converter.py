# Converts the tif images to jpg images in the ak folder.

import os
import cv2
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm

MASKS = False

print("Flushing directories...")
if MASKS:
    if not os.path.exists("./ak-dataset/masks_msr_cropped_jpeg"):
        os.mkdir("./ak-dataset/masks_msr_cropped_jpeg")
        os.mkdir("./ak-dataset/masks_msr_cropped_jpeg/0_positive")
        os.mkdir("./ak-dataset/masks_msr_cropped_jpeg/1_negative")
    shutil.rmtree("./ak-dataset/masks_msr_cropped_jpeg/0_positive")
    shutil.rmtree("./ak-dataset/masks_msr_cropped_jpeg/1_negative")
    os.mkdir("./ak-dataset/masks_msr_cropped_jpeg/0_positive")
    os.mkdir("./ak-dataset/masks_msr_cropped_jpeg/1_negative")
else:
    if not os.path.exists("./ak-dataset/ak_jpg"):
        os.mkdir("./ak-dataset/ak_jpg")
        os.mkdir("./ak-dataset/ak_jpg/0_positive")
        os.mkdir("./ak-dataset/ak_jpg/1_suspected")
        os.mkdir("./ak-dataset/ak_jpg/2_negative")
    shutil.rmtree("./ak-dataset/ak_jpg/0_positive")
    shutil.rmtree("./ak-dataset/ak_jpg/1_suspected")
    shutil.rmtree("./ak-dataset/ak_jpg/2_negative")
    os.mkdir("./ak-dataset/ak_jpg/0_positive")
    os.mkdir("./ak-dataset/ak_jpg/1_suspected")
    os.mkdir("./ak-dataset/ak_jpg/2_negative")
print("Done.")

if MASKS:
    image_path = "./ak-dataset/masks_cropped"
else:
    image_path = "./ak-dataset/ak"

categories = glob(image_path+'/*')
filedf = pd.DataFrame()
for cat in categories:
    files = glob(cat+'/*')
    tempdf = pd.DataFrame({'filepath' : files, 'category' : cat.split("/")[-1]})
    filedf = pd.concat([tempdf,filedf])

print(filedf.iloc[0]['filepath'])
print(filedf.iloc[0]['category'])

# Convert tif to jpeg
for i,row in tqdm(filedf.iterrows()):
    inpath = row['filepath']
    if MASKS:
        outpath = inpath.replace("ak-dataset/masks_cropped", "ak-dataset/masks_msr_cropped_jpeg").replace(".tif", ".jpg")
    else:
        outpath = inpath.replace("ak-dataset/ak", "ak-dataset/ak_jpg").replace(".tif", ".jpg")
    image = cv2.imread(inpath)
    cv2.imwrite(outpath, image)