import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

paths = {'id_path': 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt',
         'data_path': 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/',
         'label_path': 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/'}

image_ids = open(paths['id_path'], 'r').read().split('\n')
train_ids, test_ids = train_test_split(image_ids, test_size=0.33, random_state=2018)
print('Total data size: {}\nTraining data size: {}\nTest data size: {}\n'
      .format(len(image_ids), len(train_ids), len(test_ids)))

n_labels = 21
x_train = []
y_train = []
x_test = []
y_test = []

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

for file in os.listdir(paths['data_path']):
    if file.split('.')[0] in train_ids:
        image = Image.open(paths['data_path'] + file).convert('RGB')
        x_train.append(np.asarray(image))
    elif file.split('.')[0] in test_ids:
        image = Image.open(paths['data_path'] + file).convert('RGB')
        x_test.append(np.asarray(image))

palette = pascal_palette()
for file in os.listdir(paths['label_path']):
    if file.split('.')[0] in train_ids:
        image = np.asarray(Image.open(paths['label_path'] + file).convert('RGB'))
        image = np.reshape(image, (-1, 3))
        new_image = np.zeros([np.shape(image)[0], n_labels])
        for i in range(len(image)):
            if tuple(image[i]) in palette.keys():
                new_image[i, palette[tuple(image[i])]] = 1
        y_train.append(new_image)
    elif file.split('.')[0] in test_ids:
        image = np.asarray(Image.open(paths['label_path'] + file).convert('RGB'))
        image = np.reshape(image, (-1, 3))
        new_image = np.zeros([np.shape(image)[0], n_labels])
        for i in range(len(image)):
            if tuple(image[i]) in palette.keys():
                new_image[i, palette[tuple(image[i])]] = 1
        y_test.append(new_image)
print(y_train.shape)
