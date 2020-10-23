import argparse
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


def prepare_datasets(label):
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data(path=data_filepath+'mnist.npz')
    data = list(filter(lambda v: v[1] == label, zip(x, y)))
    (xt, yt) = list(zip(*data))
    train_labels = np.array(yt)
    train_images = np.array(xt)
    return train_images, train_labels


def main(epochs):
    for label in labels:
        train_label_gan(label, epochs)


def train_label_gan(label, epochs):
    print('Train GAN for {} label'.format(label))
    train_images, train_labels = prepare_datasets(label)
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
    parser.add_argument('--epochs', default=100)
    args = parser.parse_args()
    epochs = args.epochs
    main(epochs)
