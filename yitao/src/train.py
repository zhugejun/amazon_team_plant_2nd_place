# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np
import pandas as pd
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model

from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time
import  sys
from multi_gpu import make_parallel
from densenet169 import densenet169_model

import tensorflow as tf
import  pickle
tf.logging.set_verbosity(tf.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")
x_train = []
y_train = []
nfolds = 5

df_train = pd.read_csv('../input/train_v2.csv')



train_ids = np.array(df_train['image_name'])

print  "train_ids.shape ", train_ids.shape

labels = ['blow_down',
          'bare_ground',
          'conventional_mine',
          'blooming',
          'cultivation',
          'artisinal_mine',
          'haze',
          'primary',
          'slash_burn',
          'habitation',
          'clear',
          'road',
          'selective_logging',
          'partly_cloudy',
          'agriculture',
          'water',
          'cloudy']

label_map = {'agriculture': 14,
             'artisinal_mine': 5,
             'bare_ground': 1,
             'blooming': 3,
             'blow_down': 0,
             'clear': 10,
             'cloudy': 16,
             'conventional_mine': 2,
             'cultivation': 4,
             'habitation': 9,
             'haze': 6,
             'partly_cloudy': 13,
             'primary': 7,
             'road': 11,
             'selective_logging': 12,
             'slash_burn': 8,
             'water': 15}


aug_type = 4
tag = sys.argv[1]
image_size = int(sys.argv[2])
network = sys.argv[3]
single_batch_size = 32

print  " \n\n ****************   "
print  " training   "

print  " tag = ", tag
print  " image_size = ", image_size
print  " network = ", network
print  " aug_type = ", aug_type
print  " single_batch_size = ", single_batch_size



num_fold = 0
sum_score = 0


cv_id_list=[]
cv_pred_list = []

new_train_id_array = pickle.load(open('../data/new_train_id_array' + str(aug_type) + '_' + str(image_size),'rb'))

kf = KFold(len(new_train_id_array), n_folds=nfolds, shuffle=True, random_state=1)

score_list = []
for train_index, test_index in kf:
    start_time_model_fitting = time.time()

    num_fold += 1


    redidu = len(train_index)%4
    if redidu > 0:
        train_index = train_index[:len(train_index)- redidu]
        assert  len(train_index)%4 == 0

    redidu = len(test_index)%4
    if redidu > 0:
        test_index = test_index[:len(test_index)- redidu]
        assert  len(test_index)%4 == 0



    X_train = np.load('../data/X_train_aug'+ str(aug_type) + '_' + str(image_size) +'_' +str(num_fold) +'.npy')
    Y_train = np.load('../data/Y_train_aug'+ str(aug_type) + '_' + str(image_size) +'_' +str(num_fold)+'.npy')
    X_valid = np.load('../data/X_valid_aug'+ str(aug_type) + '_' + str(image_size) +'_' +str(num_fold)+'.npy')
    Y_valid = np.load('../data/Y_valid_aug'+ str(aug_type) + '_' + str(image_size) +'_' +str(num_fold)+'.npy')

    id_valid = [new_train_id_array[i] for i in test_index]

    print X_train.shape
    print X_valid.shape

    print Y_train.shape
    print Y_valid.shape


    print('Start KFold number {} from {}'.format(num_fold, nfolds))
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))

    kfold_weights_path = os.path.join('../cv/', tag+ '_weights_kfold_' + str(num_fold) + '.h5')

    epochs_arr = [2, 20]
    learn_rates = [0.0001, 0.00005]
    initial_model = []
    if network == 'Xception':
        print "using Xception "
        initial_model = Xception(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    if network == 'InceptionV3':
        learn_rates = [0.001, 0.0001]
        initial_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

    if network == 'DenseNet':
        initial_model = densenet169_model(image_size,image_size)

    if network == 'VGG19':
        initial_model = VGG19(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

    if network == 'VGG16':
        initial_model = VGG16(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

    if network == 'ResNet50':
        initial_model = ResNet50(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))


    last = initial_model.output
    x = Flatten()(last)
    x = Dense(512, activation='relu')(x)

    x = Dense(1024, activation='relu')(x)
    preds = Dense(17, activation='sigmoid')(x)
    model = Model(initial_model.input, preds)
    model = make_parallel(model, 4)


    print 'learn_rates: ', learn_rates
    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        opt = optimizers.Adam(lr=learn_rate)
        model.compile(loss='binary_crossentropy',
                      # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                      optimizer=opt,
                      metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=0, verbose=0),
                     ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

        model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=single_batch_size*4, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)

    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    p_valid = model.predict(X_valid, batch_size=32, verbose=2)

    cv_id_list.extend(id_valid)
    cv_pred_list.extend(p_valid)

    pickle.dump(id_valid, open('../data/id_valid_' + str(tag)  +'_' +str(num_fold), 'wb'))
    pickle.dump(p_valid, open('../data/p_valid_' + str(tag)  +'_' +str(num_fold), 'wb'))

    f_score = fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
    score_list.append(f_score)
    print(f_score)
    print '\n\n *************\n\n'
    del X_train
    del X_valid
    del Y_train
    del Y_valid

print  "cv_pred_list : ", len(cv_pred_list)
print  "cv_id_list : ", len(cv_id_list)

cv_out_frame = pd.DataFrame(cv_pred_list, columns=labels)
cv_out_frame['image_name'] = cv_id_list

cv_out_frame = cv_out_frame.sort_values(by='image_name', ascending=True)
cv_out_frame.to_csv('../cv/'+ str(tag)+'_cv.csv', index=False)

print "avg scor ::: ", np.mean(score_list)
