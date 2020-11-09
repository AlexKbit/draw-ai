import argparse
import torch
import tensorflow as tf
from utils.train_utils import build_model, fit_model, evaluate_model


data_filepath = 'datasets'

# Hyperparameters for our network
model_path = 'service/models/model.nnet'
input_size = 784
hidden_sizes = [392, 196, 98]
output_size = 10
dropout = 0.0
# Fit parameters
n_chunks = 1000
learning_rate = 0.003
weight_decay = 0


def prepare_datasets():
    """
    Take data from examples and split to train.
    :return: X_train, X_test, y_train, y_test
    """
    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=data_filepath + 'mnist.npz')
    x = x.reshape(x.shape[0], input_size) / 255
    x_test = x_test.reshape(x_test.shape[0], input_size) / 255
    return x, y, x_test, y_test


def main(epochs):
    print('Start train process with below properties:')
    print('Train epochs: {}'.format(epochs))
    X_train, X_test, y_train, y_test = prepare_datasets()

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
    metainfo = {'input_size': input_size,
                'output_size': output_size,
                'hidden_layers': hidden_sizes,
                'dropout': dropout,
                'state_dict': model.state_dict()}
    print('End fit')
    torch.save(metainfo, model_path)
    print("Model saved to {}\n".format(model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25)
    args = parser.parse_args()
    epochs = args.epochs
    main(epochs)
