# import helpers
import numpy as np
import pandas as pd
import os
from os import path
import pickle
import random
import wget

# import PIL for image manipulation
from PIL import Image
from PIL import ImageOps
from sklearn.model_selection import train_test_split


def download_data(categories, data_filepath = '../datasets'):
    """
    Download data and save to data_filepath.
    :param categories:  (list) list of download image categories
    :param data_filepath: (str) path to data
    """
    # URL to dataset in GCP Storage
    dataset_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for category in categories:
        if not os.path.exists(data_filepath + '/' + str(category) + '.npy'):
            print("Start downloading data process for [{}].".format(category))
            url = dataset_url + str(category) + '.npy'
            wget.download(
                url=url,
                out=data_filepath
            )
            print("Dataset for {} was successfully downloaded.".format(category))
        else:
            print("Dataset for {} is already downloaded.".format(category))


def to_label_dict(categories, data_filepath = '../datasets'):
    """
    Convert list of caterogies to label dict with metainformation about data files.
    :param categories:  (list) list of download image categories
    :param data_filepath: (str) path to data
    :return: dict
    """
    classes_dict = {}
    for category in categories:
        classes_dict[category] = np.load(data_filepath + '/' + str(category) + '.npy')
    return classes_dict


def prepare_data(classes_dict):
    """
    Generate labels and add labels to loaded data
    :param classes_dict: (dict) dict of labels
    :return: X, y
    """
    for i, (key, value) in enumerate(classes_dict.items()):
        value = value.astype('float32') / 255.
        if i == 0:
            classes_dict[key] = np.c_[value, np.zeros(len(value))]
        else:
            classes_dict[key] = np.c_[value, i * np.ones(len(value))]

    lst = []
    for key, value in classes_dict.items():
        lst.append(value[:3000])
    tmp = np.concatenate(lst)

    # Split the data into features and class labels (X & y respectively)
    y = tmp[:, -1].astype('float32')
    X = tmp[:, :784]
    return X, y


def save_np_data(data, file_name, data_filepath = '../datasets'):
    """
    Save numpy array as stage
    :param data: (numpy array) dataset
    :param file_name: file name
    :param data_filepath: path to save
    """
    with open('{}/{}'.format(data_filepath, file_name), 'wb') as f:
        np.save(f, data)


def load_np_data(file_name, data_filepath = '../datasets'):
    """
    Load numpy array from stage
    :param file_name: file name
    :param data_filepath: path to save
    :return: (numpy array) dataset
    """
    with open('{}/{}'.format(data_filepath, file_name), 'rb') as f:
        return np.load(f)


def convert_to_PIL(img):
    """
    Convert numpy array (1, 784) image to PIL image.
    :param img: (numpy array) image from train dataset with size (1, 784)
    :return: (PIL Image) 28x28 image
    """
    img_r = img.reshape(28,28)

    pil_img = Image.new('RGB', (28, 28), 'white')
    pixels = pil_img.load()

    for i in range(0, 28):
        for j in range(0, 28):
            if img_r[i, j] > 0:
                pixels[j, i] = (255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255))

    return pil_img


def convert_to_np(pil_img):
    """
    Convert PIL Image to numpy array.
    :param pil_img: (PIL Image) 28x28 image to be converted
    :return: (numpy array) converted image with shape (28, 28)
    """
    pil_img = pil_img.convert('RGB')

    img = np.zeros((28, 28))
    pixels = pil_img.load()

    for i in range(0, 28):
        for j in range(0, 28):
            img[i, j] = 1 - pixels[j, i][0] / 255

    return img


def rotate_image(src_im, angle = 45, size = (28,28)):
    """
    Rotate PIL Image file
    :param src_im: (PIL Image) 28x28 image to be rotated
    :param angle: angle to rotate the image
    :param size: (tuple) size of the output image
    :return: (PIL Image) rotated image
    """
    dst_im = Image.new("RGBA", size, "white")
    src_im = src_im.convert('RGBA')

    rot = src_im.rotate(angle)
    dst_im.paste(rot, (0, 0), rot)

    return dst_im


def flip_image(src_im):
    """
    Flip a PIL Image file.
    :param src_im: (PIL Image) 28x28 image to be flipped
    :return: (PIL Image) flipped image
    """
    dst_im = src_im.transpose(Image.FLIP_LEFT_RIGHT)
    return dst_im


def join_transformed_images(X_train, y_train):
    """
    Function which adds flipped and rotated images to the original dataset.
    :param X_train: (numpy array) the original training set
    :param y_train: (numpy array) original labels dataset
    :return: X_train_new, y_train_new
    """
    print("Adding flipped and rotated images to the training set. \n")

    X_train_new = X_train.copy()
    y_train_new = y_train.copy().reshape(y_train.shape[0], 1)

    for i in range(0, X_train.shape[0]):
        # get image to rotate and flip
        img = X_train[i]
        pil_img = convert_to_PIL(img)

        # get random angle
        angle = random.randint(5, 10)

        # rotate and flip
        rotated = convert_to_np(rotate_image(pil_img, angle))
        flipped = convert_to_np(flip_image(pil_img))

        # add to the original dataset
        X_train_new = np.append(X_train_new, rotated.reshape(1, 784), axis = 0)
        X_train_new = np.append(X_train_new, flipped.reshape(1, 784), axis = 0)
        y_train_new = np.append(y_train_new, y_train[i].reshape(1,1), axis = 0)
        y_train_new = np.append(y_train_new, y_train[i].reshape(1,1), axis = 0)

        # print out progress
        if i % 100 == 0:
            print("Processed {i} files out of {total}.".format(i= i, total = X_train.shape[0]))

    return X_train_new, y_train_new


