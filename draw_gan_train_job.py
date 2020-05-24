import argparse
import wget
import numpy as np
import os
import json
import tensorflow as tf

from utils.gan_utils import make_generator_model, make_discriminator_model, train


def load_label_dict():
    with open('service/label_dict.json') as f:
        return json.load(f)


# URL to dataset in GCP Storage
dataset_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
labels = load_label_dict()
data_filepath = 'datasets'
num_categories = len(labels)
BUFFER_SIZE = 60000
BATCH_SIZE = 256

noise_dim = 100
num_examples_to_generate = 16


def download_datasets(labels):
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
    classes_dict = {}
    for category in labels:
        classes_dict[category] = np.load(data_filepath + '/' + str(category) + '.npy')
    # Generate labels and add labels to loaded data
    for i, (key, value) in enumerate(classes_dict.items()):
        value = value.astype('float32')
        if i == 0:
            classes_dict[key] = np.c_[value, np.zeros(len(value))]
        else:
            classes_dict[key] = np.c_[value, i * np.ones(len(value))]

    lst = []
    for key, value in classes_dict.items():
        lst.append(value[:num_examples])
    tmp = np.concatenate(lst)

    # Split the data into features and class labels (X & y respectively)
    y = tmp[:, -1].astype('int')
    x = tmp[:, :784]
    return x, y


def main(num_examples, epochs):
    download_datasets(labels)
    for label in labels:
        train_label_gan(label, num_examples, epochs)


def train_label_gan(label, num_examples, epochs):
    print('Train GAN for {} label'.format(label))
    train_images, train_labels = prepare_datasets(list([label]), num_examples)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    train(train_dataset, noise_dim, epochs, BATCH_SIZE, generator, discriminator,
          generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix, cross_entropy)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    model_path = 'service/models/{}-gan.h5'.format(label)
    generator.save(model_path)
    print('Gan for {} saved to {}'.format(label, model_path))
    generator.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', default=6000)
    parser.add_argument('--epochs', default=100)
    args = parser.parse_args()
    num_examples = args.num_examples
    epochs = args.epochs
    main(num_examples, epochs)