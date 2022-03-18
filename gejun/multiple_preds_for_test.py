#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:34:34 2017

@author: gejun
"""
import numpy as np
np.random.seed(2016920)
import pandas as pd
import os
import settings
import pprint

from keras.models import Sequential, Model
from keras.applications import VGG16
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Input, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from glob import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import f2_score_k, f2_score
from utils import optimise_f2_thresholds

from keras.layers.merge import concatenate
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from scale_layer import Scale
import keras.backend as K


from sys import argv
num_folder = argv[1]


print('-'*40)
print('training on floder: ', str(num_folder))

BATCH_SIZE = 32 #changed from 64
TARGET_SIZE = (224, 224)

#----------------------------------#

labels = np.array(['primary', 'clear', 'agriculture', 'road', 'water',
  'partly_cloudy', 'cultivation', 'habitation', 'haze',
  'cloudy', 'bare_ground', 'selective_logging', 'artisinal_mine',
  'blooming', 'slash_burn', 'blow_down', 'conventional_mine'])

n_classes = len(labels)

label_map = {'conventional_mine': 16,
    'blow_down': 15,
    'slash_burn': 14,
    'blooming': 13,
    'artisinal_mine': 12,
    'selective_logging': 11,
    'bare_ground': 10,
    'cloudy': 9,
    'haze': 8,
    'habitation': 7,
    'cultivation': 6,
    'partly_cloudy': 5,
    'water': 4,
    'road': 3,
    'agriculture': 2,
    'clear': 1,
    'primary': 0}

inv_label_map = {i: l for l, i in label_map.items()}
pprint.pprint(label_map)


#----------model part----------#

def densenet121_model(img_rows, img_cols, color_type=1, nb_dense_block=4,
  growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    DenseNet 121 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx],
          nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression,
          dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate,
      dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    weights_path = settings.WEIGHT_DIR + 'densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('sigmoid', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate,
  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

#    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter



#---------------------------------#

def rotate_img(img, rotate_angle):
    if rotate_angle == 0:
        return img
    else:
        rows, cols, _ = img.shape

        img1 = cv2.getRotationMatrix2D((cols/2,rows/2), rotate_angle, 1)
        img1 = cv2.warpAffine(img, img1, (cols, rows))
        return img1

def get_img(img_file, rotate_angle):
    img = cv2.imread(img_file)
    img = rotate_img(img, rotate_angle)
    img = cv2.resize(img, TARGET_SIZE)
    return img


def load_test_images(rotate_angle):
    x_test = []
    image_names = []
    for f in tqdm(sorted(glob(settings.TEST_IMG_DIR + '/*.jpg')), miniters=1000):
        image_names.append(os.path.basename(f).split('.')[0])
        img = get_img(f, rotate_angle)
        x_test.append(img)
    print('normalizing data...')
    X_test = np.array(x_test, np.float16) / 255.0
    return image_names, X_test


# threshold_nf = np.load(settings.OOF_DIR + 'thresholds_' + str(num_folder) + '.npy')

val_f2 = glob(settings.OOF_DIR + 'oof_val_' + str(num_folder) + '*.csv')[0].split('_')[-1].replace('.csv', '')

model = densenet121_model(TARGET_SIZE[0], TARGET_SIZE[1], color_type=3, num_classes=17)
kfold_weight_path = settings.WEIGHT_DIR + 'keras_densent121_' + str(num_folder) + '.h5'

if os.path.isfile(kfold_weight_path):
    print('Weights loaded.')
    model.load_weights(kfold_weight_path)


# 4 different angles

angles = [0, 90, 180, 270]
y_test_prob = np.zeros((61191, 17))
for angle in angles:
    print('-'*40 + str(angle) + '-'*40)
    print('loading test data...')
    image_names, X_test = load_test_images(angle)

    print('making predictions...')
    y_test_prob += model.predict(X_test, verbose = 1)

y_test_prob = y_test_prob / len(angles)


# saving predictions probability
df_test_prob = pd.DataFrame(y_test_prob, columns = labels)
df_test_prob['image_name'] = image_names
df_test_prob.to_csv(settings.OOF_DIR + 'oof_test_prob_' + str(num_folder) + '_' + str(val_f2) + '.csv', index=False)

print('*********************** Folder ' + str(num_folder) + ' done.*****************************')










