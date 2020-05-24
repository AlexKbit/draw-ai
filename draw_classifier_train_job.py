import numpy as np
import os
import argparse
import wget
import json

import torch
from sklearn.model_selection import train_test_split

from utils.image_utils import join_transformed_images
from utils.train_utils import build_model, fit_model, evaluate_model


def load_label_dict():
    with open('service/label_dict.json') as f:
        return json.load(f)

# URL to dataset in GCP Storage
dataset_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
labels = load_label_dict()
data_filepath = 'datasets'
num_categories = len(labels)

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 100, 64]
output_size = 10
dropout = 0.0
# Fit parameters
n_chunks = 1000
learning_rate = 0.003
weight_decay = 0


def download_datasets(labels):
    """
    Download data for each label
    :param labels: list of labels
    """
    for category in labels:
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


def prepare_datasets(labels, num_examples):
    """
    Take some number of data from examples and split to train.
    :param labels: list of labels
    :param num_examples: number of examples
    :return: X_train, X_test, y_train, y_test
    """
    classes_dict = {}
    for category in labels:
        classes_dict[category] = np.load(data_filepath + '/' + str(category) + '.npy')
    # Generate labels and add labels to loaded data
    for i, (key, value) in enumerate(classes_dict.items()):
        value = value.astype('float32') / 255.
        if i == 0:
            classes_dict[key] = np.c_[value, np.zeros(len(value))]
        else:
            classes_dict[key] = np.c_[value, i * np.ones(len(value))]

    lst = []
    for key, value in classes_dict.items():
        lst.append(value[:num_examples])
    tmp = np.concatenate(lst)

    # Split the data into features and class labels (X & y respectively)
    y = tmp[:, -1].astype('float32')
    X = tmp[:, :784]

    # Split each dataset into train/test splits
    return train_test_split(X, y, test_size=0.3, random_state=1)


def main(num_examples, epochs):
    print('Start train process with below properties:')
    print('Number of examples: {}'.format(num_examples))
    print('Train epochs: {}'.format(epochs))
    download_datasets(labels)
    X_train, X_test, y_train, y_test = prepare_datasets(labels, num_examples)
    print('Generate new data and join to train dataset')
    X_train, y_train = join_transformed_images(X_train, y_train)

    train = torch.from_numpy(X_train).float()
    train_labels = torch.from_numpy(y_train).long()
    test = torch.from_numpy(X_test).float()
    test_labels = torch.from_numpy(y_test).long()

    print('Build model')
    model = build_model(input_size, output_size, hidden_sizes, dropout)
    print('Start fitting')
    fit_model(model, train, train_labels,
              epochs=epochs, n_chunks=n_chunks, learning_rate=learning_rate, weight_decay=weight_decay)
    evaluate_model(model, train, train_labels, test, test_labels)
    filepath = '../service/models/model.nnet'
    metainfo = {'input_size': input_size,
                'output_size': output_size,
                'hidden_layers': hidden_sizes,
                'dropout': dropout,
                'state_dict': model.state_dict()}
    print('End fit')
    torch.save(metainfo, filepath)
    print("Model saved to {}\n".format(filepath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', default=3000)
    parser.add_argument('--epochs', default=100)
    args = parser.parse_args()
    num_examples = args.num_examples
    epochs = args.epochs
    main(num_examples, epochs)