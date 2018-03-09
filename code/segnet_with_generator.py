import os
import numpy as np
from PIL import Image
from keras import models
from keras.models import load_model
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import MaxPooling2D, UpSampling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split

n_labels = 21
kernel = 3
pool_size = 2
paths = {'id_path': 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt',
             'data_path': 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/',
             'label_path': 'SegmentationClassBboxSeg_Visualization/SegmentationClassBboxErode20CRFAug_Visualization/'}


def build_model():
    encoding_layers = [
        Conv2D(64, (kernel, kernel), padding='same', input_shape=(None, None, 3)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

    decoding_layers = [
        UpSampling2D(size=(pool_size, pool_size)),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size, pool_size)),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size, pool_size)),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size, pool_size)),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size, pool_size)),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(n_labels, (1, 1), padding='valid'),
        BatchNormalization(),
    ]


    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
        print(l.input_shape,l.output_shape,l)

    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.add(Reshape((-1, n_labels)))
    autoencoder.add(Activation('softmax'))

    autoencoder.summary()
    return autoencoder


def pascal_classes():
    classes = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
               'bottle': 5, 'bus': 6,  'car': 7, 'cat': 8,
               'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12,
               'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
               'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}
    return classes


def pascal_palette():
    palette = {(0, 0, 0): 0,
               (128, 0, 0): 1,
               (0, 128, 0): 2,
               (128, 128, 0): 3,
               (0, 0, 128): 4,
               (128, 0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64, 0, 0): 8,
               (192, 0, 0): 9,
               (64, 128, 0): 10,
               (192, 128, 0): 11,
               (64, 0, 128): 12,
               (192, 0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0, 64, 0): 16,
               (128, 64, 0): 17,
               (0, 192, 0): 18,
               (128, 192, 0): 19,
               (0, 64, 128): 20}
    return palette


def prepare_data():
    image_ids = open(paths['id_path'], 'r').read().split('\n')
    train_ids, test_ids = train_test_split(image_ids, test_size=0.33, random_state=2018)
    print('Total data size: {}\nTraining data size: {}\nTest data size: {}\n'
          .format(len(image_ids), len(train_ids), len(test_ids)))
    return train_ids, test_ids


def data_generator(image_ids):
    x_data = {}
    y_data = {}

    print('Loading data ... ')
    for file in os.listdir(paths['data_path']):
        id = file.split('.')[0]
        if id in image_ids:
            image = Image.open(paths['data_path'] + file).convert('RGB')
            x_data[id] = np.asarray(image)

    for file in os.listdir(paths['label_path']):
        id = file.split('.')[0]
        if file.split('.')[0] in image_ids:
            image = np.asarray(Image.open(paths['label_path'] + file).convert('RGB'))
            y_data[id] = image
    print('Data loaded !')

    palette = pascal_palette()
    while True:
        for key in x_data.keys():
            print('Training image: {} -------- Image size: {} -------- Same size: {}'
                  .format(key, np.shape(x_data[key]), np.shape(x_data[key]) == np.shape(y_data[key])))
            x_image = np.expand_dims(x_data[key], axis=0)
            y_image = y_data[key]
            y_image = np.reshape(y_image, (-1, 3))
            y_new_image = np.zeros([np.shape(y_image)[0], n_labels])
            for j in range(len(y_image)):
                if tuple(y_image[j]) in palette.keys():
                    y_new_image[j, palette[tuple(y_image[j])]] = 1
            y_new_image = np.expand_dims(y_new_image, axis=0)
            yield x_image, y_new_image


if __name__ == '__main__':
    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

    if os.path.exists("segnet.h5"):
        load_model('segnet.h5')
    else:
        train_ids, test_ids = prepare_data()
        train_gen = data_generator(train_ids)
        #valid_gen = data_generator(test_ids)
        model.fit_generator(train_gen, steps_per_epoch=len(train_ids), epochs=1, verbose=2)
                            #validation_data=valid_gen, validation_steps=len(test_ids))
        model.save('segnet.h5')

