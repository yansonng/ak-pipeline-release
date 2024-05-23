import os
import random

def get_patient_info_one_class(dir_one_class):
    '''
    get the patient wise information from the directory
    Parameters:
        dir (str): the directory path, should end with "/0_positive" or "/2_negative"
    Returns:
        patient_info_lst (dict): dictionary containing patient wise information for one class
    '''
    # iterate through the folder, and store info in a list, store it in patient_info_lst
    # patient id -> the first parameter in path split by '_'
    # patient class -> positive or negative from the folder name
    # number of images -> number of images having patient id as first parameter in the folder

    # dictionary to store patient id, patient class, number of images
    # key: patient id, value: [patient id, patient class, number of images]
    patient_info_lst = {}

    # iterate through the positive and negative dir
    for file in os.listdir(dir_one_class):
      # store the patient id as key
      patient_id = file.split('_')[0]
      patient_class = dir_one_class.split('_')[-1]
      # if the patient id is not in the dictionary, and it is a number, add it
      if patient_id not in patient_info_lst and patient_id.isdigit():
            patient_info_lst[patient_id] = [patient_id, patient_class, 1]
      else:
            if patient_id.isdigit():
                patient_info_lst[patient_id][2] += 1

    return patient_info_lst


def get_patient_info(pos_dir, neg_dir):
    '''
    Function to get the patient wise information from the positive and negative directory
    Parameters:
        pos_dir (str): the positive directory
        neg_dir (str): the negative directory
    Returns:
        patient_info (dict): dictionary containing patient wise information
    '''
    # get the patient wise information from the positive and negative directory
    pos_patient_info = get_patient_info_one_class(pos_dir)
    neg_patient_info = get_patient_info_one_class(neg_dir)

    # merge two dictionaries
    patient_info = {**pos_patient_info, **neg_patient_info}

    return patient_info



def get_patient_img_path(patient_id, pos_dir, neg_dir, absolute_path=False):
    '''
    Function to get the image path of a patient
    Parameters:
        patient_id (str): the id of the patient
        pos_dir (str): the positive directory
        neg_dir (str): the negative directory
        absolute_path (bool): if True, return the absolute path of the image, else return the relative path
    Returns:
        patient_img_path (list): list of image paths of the patient
    '''
    patient_img_path = []
    # iterate through the positive and negative dir
    # if the patient id is in first parameter of the file name, add it to the list

    for file in os.listdir(pos_dir):
        file_patient_id = file.split('_')[0]
        if file_patient_id == patient_id:
            if absolute_path:
                patient_img_path.append(os.path.abspath(os.path.join(pos_dir, file)))
            else:
                patient_img_path.append(os.path.join(neg_dir, file))

    for file in os.listdir(neg_dir):
        file_patient_id = file.split('_')[0]
        if file_patient_id == patient_id:
            if absolute_path:
                    patient_img_path.append(os.path.abspath(os.path.join(neg_dir, file)))
            else:
                patient_img_path.append(os.path.join(neg_dir, file))

    return patient_img_path

def train_test_val_split(train_ratio, test_ratio, val_ratio, pos_dir, neg_dir):
    '''
    Function to split the data into train, test, and validation set
    Parameters:
        train_ratio (float): the ratio of training data
        test_ratio (float): the ratio of testing data
        val_ratio (float): the ratio of validation data (Temporarily unused)
        pos_dir (str): the positive directory
        neg_dir (str): the negative directory
    Returns:
        train_set (list): list of training data
        test_set (list): list of testing data
        val_set (list): list of validation data
    '''

    patient_info_dict = get_patient_info(pos_dir, neg_dir)

    # Shuffle patient_info into list
    patient_info = list(patient_info_dict.values())
    random.shuffle(patient_info)

    total_images_count = 0
    for patient in patient_info:
        total_images_count += patient[2]

    train_count = int(total_images_count * train_ratio)
    test_count = int(total_images_count * test_ratio)
    val_count = int(total_images_count * val_ratio)
    train_paths = []
    test_paths = []
    val_paths = []

    for patient in patient_info:
        patient_id = patient[0]
        patient_img_paths = get_patient_img_path(str(patient_id), pos_dir, neg_dir, absolute_path=True)
        if len(train_paths) < train_count:
            train_paths.extend(patient_img_paths)
        elif len(val_paths) < val_count:
            val_paths.extend(patient_img_paths)
        else:
            test_paths.extend(patient_img_paths)

    return train_paths, test_paths, val_paths

def k_fold_split(k, pos_dir, neg_dir, verbose=False):
    '''
    Function to split the data into k folds patient wise
    Parameters:
        k (int): the number of folds
        pos_dir (str): the positive directory
        neg_dir (str): the negative directory
    Returns:
        k_folds (list): list of k folds
    '''
    patient_info_dict = get_patient_info(pos_dir, neg_dir)

    # Shuffle patient_info into list
    patient_info = list(patient_info_dict.values())

    total_images_count = 0
    for patient in patient_info:
        total_images_count += patient[2]

    fold_size = total_images_count // k


    while True:
        random.shuffle(patient_info)
        k_folds = [[] for i in range(k)]
        fold_lengths_remaining = [fold_size for i in range(k)]
        for patient_idx in range(len(patient_info)):
            max_fold_index = fold_lengths_remaining.index(max(fold_lengths_remaining))
            fold_lengths_remaining[max_fold_index] -= patient_info[patient_idx][2]
            k_folds[max_fold_index].append(get_patient_img_path(str(patient_info[patient_idx][0]), pos_dir, neg_dir, absolute_path=True))

        if verbose:
            print(fold_lengths_remaining, max(fold_lengths_remaining) - min(fold_lengths_remaining), end="\r")

        if max(fold_lengths_remaining) - min(fold_lengths_remaining) <= 50:
            break

    if verbose:
        for i in range(len(k_folds)):
                print(f"Fold {i+1}: From {len(k_folds[i])} patients, total {sum([len(patient) for patient in k_folds[i]])} images")
    return k_folds

if __name__ == "__main__":
    # example usage
    pos_dir = "./ak-dataset/ak_jpg/0_positive"
    neg_dir = "./ak-dataset/ak_jpg/2_negative"


    # patient_info = get_patient_info(pos_dir, neg_dir)
    # print(patient_info)

    # patient_9_imgs = get_patient_img_path("9", pos_dir, neg_dir, absolute_path=True)
    # print(patient_9_imgs)

    k_fold_paths = k_fold_split(5, pos_dir, neg_dir, verbose=True)
