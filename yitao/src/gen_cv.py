# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from tqdm import tqdm

from skimage import transform
from sklearn.cross_validation import KFold
import time
import  sys
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
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





import  random
def randomRotate90(img,angle, u=1):
    if random.random() < u:
        if angle == 90:
            img = cv2.transpose(img)
            img = cv2.flip(img,1)
        elif angle == 180:
            img = cv2.flip(img,-1)
        elif angle == 270:
            img = cv2.transpose(img)
            img = cv2.flip(img,0)
    return img



print  " \n\n ****************   "
print  " gen_cv   "

image_size = int(sys.argv[1])

print  " image_size = ", image_size

aug_type = 4

print  " aug_type = ", aug_type


new_train_id_array = []
for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    img = cv2.resize(img, (image_size, image_size))


    if aug_type == 4:

        x_train.append(img)
        y_train.append(targets)
        new_train_id_array.append(f)

        x_train.append(randomRotate90(img, 90))
        y_train.append(targets)
        new_train_id_array.append(f)

        x_train.append(randomRotate90(img, 180))
        y_train.append(targets)
        new_train_id_array.append(f)

        x_train.append(randomRotate90(img, 270))
        y_train.append(targets)
        new_train_id_array.append(f)


kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)

pickle.dump(new_train_id_array,open('../data/new_train_id_array' + str(aug_type) + '_' + str(image_size),'wb'))

num_fold = 0
sum_score = 0

cv_id_list=[]
cv_pred_list = []

score_list = []
for train_index, test_index in kf:

    num_fold += 1

    start_time_model_fitting = time.time()

    #Using  tensor flow multi_gpu training, the data size must be n*n_gpus
    redidu = len(train_index)%4
    if redidu > 0:
        train_index = train_index[:len(train_index)- redidu]
        assert  len(train_index)%4 == 0

    redidu = len(test_index)%4
    if redidu > 0:
        test_index = test_index[:len(test_index)- redidu]
        assert  len(test_index)%4 == 0


    X_train = [x_train[i] for i in train_index]
    Y_train = [y_train[i] for i in train_index]
    X_valid = [x_train[i] for i in test_index]
    Y_valid = [y_train[i] for i in test_index]
    id_valid = [new_train_id_array[i] for i in test_index]


    X_train = np.array(X_train, np.float16) / 255.
    X_valid = np.array(X_valid, np.float16) / 255.

    Y_train = np.array(Y_train, np.uint8)
    Y_valid = np.array(Y_valid, np.uint8)

    print X_train.shape
    print X_valid.shape

    print Y_train.shape
    print Y_valid.shape

    print('Start KFold number {} from {}'.format(num_fold, nfolds))
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))

    np.save('../data/X_train_aug'+ str(aug_type) + '_' + str(image_size) +'_' +str(num_fold), X_train)
    np.save( '../data/Y_train_aug'+ str(aug_type) + '_' + str(image_size)+'_' +str(num_fold),Y_train)
    np.save( '../data/X_valid_aug'+ str(aug_type) + '_' + str(image_size)+'_' +str(num_fold),X_valid)
    np.save( '../data/Y_valid_aug' + str(aug_type) + '_' + str(image_size)+'_' +str(num_fold),Y_valid)
