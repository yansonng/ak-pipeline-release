import cv2
import pandas as pd
import os
from tqdm import tqdm

'''
This script uses the annotation csv file from the VIA tool 
to get the cx, cy coordinates of the manually picked masks.
then crop the positive images in ak-dataset/ak_jpg/0_positive to 100x100 images with the mask in the center.
The cropped images are saved in ak-dataset/ak_jpg/0_positive_cropped
'''



def get_cx_cy(csv_file, idx):
  '''
  Get the cx, cy coordinates from the row
  '''
  # get cx, cy from the row
  cx = int(csv_file.loc[idx, 'region_shape_attributes'].split(',')[1].split(':')[1])
  cy = int(csv_file.loc[idx, 'region_shape_attributes'].split(',')[2].split(':')[1])
  return cx, cy



def crop_image_with_annotation(annotation_csv_dir, ori_pos_image_dir_path, output_dir_path):
  '''
  Crop the positive images with the mask in the center
  '''
  # import annotation csvs, put all csvs in ak-dataset/ak-annotation-csv into a list
  csv_files_names = os.listdir(annotation_csv_dir)
  csv_list = []
  for csv_file in csv_files_names:
    one_csv = pd.read_csv(f'{annotation_csv_dir}/{csv_file}')
    csv_list.append(one_csv)
  

  print(f'Found {len(csv_list)} csv files')


  # put all jpeg images in ak-dataset/ak/0_positive into a list
  positive_images = os.listdir(ori_pos_image_dir_path)
  
  # iterate each csv file
  csvcount = 0
  for annotation in tqdm(csv_list, desc='Processing csv files', total=len(csv_list)):
    # for each row in csv, if filename is in positive_images, crop the image
    # iterate each row in csv
    for idx, row in annotation.iterrows():
      # get filename
      filename = annotation.loc[idx, 'filename']
      if filename in positive_images:
        # print(f'Processing {filename}')
        # if region_shape_attributes is not empty, get cx, cy from csv
        if annotation.loc[annotation['filename'] == filename, 'region_shape_attributes'].iloc[0] != '{}':
          # get cx, cy from the row
          cx, cy = get_cx_cy(annotation, idx)
          # print(f'cx: {cx}, cy: {cy}')
          # read image
          image = cv2.imread(f'{ori_pos_image_dir_path}/{filename}')
          # add border
          imagePadded = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
          # crop image
          cx += 50
          cy += 50
          cropped = imagePadded[int(cy)-50:int(cy)+50, int(cx)-50:int(cx)+50]
          # save cropped image, use idx as suffix. idx also represents the number of masks
          cv2.imwrite(f'{output_dir_path}/{filename.split(".")[0]}_{idx}' + str(csvcount) + '.jpg', cropped)
      else:
        print(f'{filename} is not in positive_images')
    csvcount += 1
  
  print('Done')





# set the paths
csv_dir_path = './via-annotation-csvs'
ori_pos_image_dir_path = './ak-dataset/ak_jpg/ak_positive_annotated'
output_dir_path = './ak-dataset/ak_jpg/0_positive_cropped'

# crop the images
crop_image_with_annotation(csv_dir_path, ori_pos_image_dir_path, output_dir_path)





