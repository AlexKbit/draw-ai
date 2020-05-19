import argparse
import base64
import io
from io import StringIO
import json
import numpy as np
from PIL import Image
from PIL import ImageOps
from skimage.io import imsave
from collections import OrderedDict

# PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Tensorflow
import tensorflow as tf
from tensorflow import keras

from flask import Flask, request, send_file

def load_label_dict():
    with open('label_dict.json') as f:
        return json.load(f)


encode_dict = load_label_dict()
model_cls = None
gan_dict = {}


def decode_class(class_index):
    return encode_dict[class_index]


def call_predict_class(session, endpoint_name, blob_img):
    response = session.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType='application/x-image',
                                       Body=blob_img)
    body = response['Body'].read().decode('utf-8')
    result = json.loads(body)
    return np.argmax(result)


app = Flask(__name__, static_url_path="", static_folder="static")


def crop_image(image):
    """
    Crops image (crops out white spaces).
    :param image: PIL image of original size to be cropped
    :return: PIL image cropped to the center  and resized to (28, 28)
    """
    cropped_image = image

    width, height = cropped_image.size
    pixels = cropped_image.load()

    image_strokes_rows = []
    image_strokes_cols = []

    for i in range(0, width):
        for j in range(0, height):
            if (pixels[i,j][3] > 0):
                image_strokes_cols.append(i)
                image_strokes_rows.append(j)

    if (len(image_strokes_rows)) > 0:
        row_min = np.array(image_strokes_rows).min()
        row_max = np.array(image_strokes_rows).max()
        col_min = np.array(image_strokes_cols).min()
        col_max = np.array(image_strokes_cols).max()

        margin = min(row_min, height - row_max, col_min, width - col_max)
        border = (col_min, row_min, width - col_max, height - row_max)
        cropped_image = ImageOps.crop(cropped_image, border)

    width_cropped, height_cropped = cropped_image.size
    dst_im = Image.new("RGBA", (max(width_cropped, height_cropped), max(width_cropped, height_cropped)), "white")
    offset = ((max(width_cropped, height_cropped) - width_cropped) // 2, (max(width_cropped, height_cropped) - height_cropped) // 2)
    dst_im.paste(cropped_image, offset, cropped_image)
    dst_im.thumbnail((28,28), Image.ANTIALIAS)
    return dst_im


def normalize(arr):
    """
    Function performs the linear normalizarion of the array.
    :param arr: original numpy array
    :return: normalized numpy array
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr


def normalize_image(image):
    """
    Function performs the normalization of the image.
    :param image: PIL image to be normalized
    :return: PIL image normalized
    """
    arr = np.array(image)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGBA')
    return new_img


def alpha_composite(front, back):
    """
    Alpha composite two RGBA images.
    :param front: PIL RGBA Image object
    :param back: PIL RGBA Image object
    :return: PIL RGBA Image object
    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result


def alpha_composite_with_color(image, color=(255, 255, 255)):
    """
    Helper function to convert RGBA to RGB.
    https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.
    :param image: PIL RGBA Image object
    :param color: Tuple RGB (default 255, 255, 255)
    :return:
    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)


def convert_to_rgb(image):
    """
    Converts RGBA PIL to RGB image.
    :param image: PIL RGBA image
    :return: PIL RGB image
    """
    image_rgb = alpha_composite_with_color(image)
    image_rgb.convert('RGB')

    return image_rgb


def load_clf_model(filepath):
    """
    Load PyTorch classification model
    :param filepath: path to model.
    :return: classificator
    """
    print("Loading model from {} \n".format(filepath))
    model_info = torch.load(filepath)
    input_size = model_info['input_size']
    output_size = model_info['output_size']
    hidden_sizes = model_info['hidden_layers']
    dropout = model_info['dropout']
    model = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                          ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('logits', nn.Linear(hidden_sizes[2], output_size))]))
    model.load_state_dict(model_info['state_dict'])
    return model


def load_gan_model(cls):
    """
    Load Tensorflow GAN to draw images.
    :param cls: class of GAN
    :return: model
    """
    return keras.models.load_model('models/{}-gan.h5'.format(cls))


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


def get_prediction(input):
    """
    Function to get prediction (label of class with the greatest probability).
    :param input: (numpy) input vector
    :return: label, label_name, preds
    """
    input = torch.from_numpy(input).float()
    input = input.resize_(1, 784)

    with torch.no_grad():
        logits = model_cls.forward(input)

    ps = F.softmax(logits, dim=1)
    preds = ps.numpy()

    label = np.argmax(preds)
    label_name = decode_class(label)

    return label, label_name, preds


def generate_image(model, img_count = 1, weight=400, height=400):
    seed = tf.random.normal([img_count, 100])
    predictions = model(seed, training=False)
    na = predictions[0].numpy()
    na = na  * 127.5 + 127.5
    img = Image.fromarray(na.reshape((28,28))).convert('RGB')
    rimg = img.resize((weight,height))
    return ImageOps.invert(rimg)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/draw')
def draw():
    return app.send_static_file('draw.html')


@app.route('/generator')
def generator():
    return app.send_static_file('generator.html')


@app.route('/fight')
def fight():
    return app.send_static_file('fight.html')


@app.route('/generate/<label>')
def generate(label):
    img = generate_image(gan_dict[label])
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/PNG')


@app.route('/classify', methods=['POST'])
def classify():
    image_b64_str = request.form['image']
    byte_data = base64.b64decode(image_b64_str.split(',')[1])
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    img = img.convert("RGBA")

    image_cropped = crop_image(img)
    image_normalized = normalize_image(image_cropped)
    img_rgb = convert_to_rgb(image_normalized)
    image_np = convert_to_np(img_rgb)

    label, label_name, preds = get_prediction(image_np)
    return json.dumps({'label': int(label),
                       'label_name': label_name,
                       'preds': json.dumps(preds.ravel().tolist())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--web_app_port', default='9000')
    args = parser.parse_args()
    print("Load classification model")
    model_cls = load_clf_model('models/model.nnet')
    model_cls.eval()
    print("Load GAN models for labels {} \n".format(encode_dict))
    for label in encode_dict:
        gan_dict[label] = load_gan_model(label)
    app.run(host='0.0.0.0', port=args.web_app_port)
