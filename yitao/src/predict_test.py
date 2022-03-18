# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm import tqdm

from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model

from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from densenet169 import densenet169_model

from multi_gpu import make_parallel

import  sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle

tag = sys.argv[1]


image_size = int(sys.argv[2])

network = sys.argv[3]

aug_type = 4

print  " \n\n ****************   "
print  " predict_test   "
print  " tag = ", tag
print  " image_size = ", image_size
print  " network = ", network

print  " aug_type = ", aug_type



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





def predict_score():

    df_train = pd.read_csv('../input/train_v2.csv')
    df_test = pd.read_csv('../input/sample_submission_v2.csv')


    train_ids = np.array(df_train['image_name'])
    test_ids = np.array(df_test['image_name'])

    print  "train_ids.shape ", train_ids.shape

    x_test = np.load('../data/X_test_aug'+ str(aug_type) + '_' + str(image_size) +'.npy')

    new_test_id_array = pickle.load(open('../data/new_test_id_array' + str(aug_type) + '_' + str(image_size), 'rb'))

    print x_test.shape
    print len(new_test_id_array)

    nfolds = 5

    num_fold = 0

    yfull_test = []


    for i in range(0,nfolds):

        num_fold += 1

        print('Start KFold number {} from {}'.format(num_fold, nfolds))

        kfold_weights_path = os.path.join('../cv/', tag+ '_weights_kfold_' + str(num_fold) + '.h5')

        initial_model = Xception(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
        if network == 'Xception':
            print "using Xception "
            initial_model = Xception(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
        if network == 'InceptionV3':
            initial_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

        if network == 'DenseNet':
            initial_model = densenet169_model(image_size, image_size)

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


        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        p_test = model.predict(x_test, batch_size=32, verbose=2)
        yfull_test.append(p_test)

    result = np.array(yfull_test[0])
    for i in range(1, nfolds):
        result += np.array(yfull_test[i])
    result /= nfolds
    result = pd.DataFrame(result, columns=labels)

    print len(new_test_id_array)

    print result.shape

    result['image_name'] = new_test_id_array

    result.to_csv('../cv/' + str(tag) + '_fulltest.csv', index=False)

    result = result.groupby(['image_name'])[labels].mean().reset_index()

    result = result.sort_values(by='image_name', ascending=True)

    sub_frame  = pd.DataFrame()
    sub_frame['image_name'] = result['image_name']

    print result.head()
    print result.shape

    del result['image_name']


    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > 0.2, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    sub_frame['tags'] = preds
    sub_frame.to_csv('../sub/' + str(tag) + '_sub_t2.csv', index=False)



def gen_sub():

    result = pd.read_csv('../cv/' + str(tag) + '_fulltest.csv')
    new_laels = labels
    new_laels.append('image_name')
    result.columns = new_laels
    print result.mean()

    result = result.groupby(['image_name'])[labels].mean().reset_index()

    result = result.sort_values(by='image_name', ascending=True)


    sub_frame = pd.DataFrame()
    sub_frame['image_name'] = result['image_name']

    print result.head()
    print result.shape

    del result['image_name']

    preds = []
    thres = np.load('../thres/thre_' + str(tag))
    print thres

    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > thres, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))

    sub_frame['tags'] = preds
    sub_frame.to_csv('../sub/' + str(tag) + '_sub_thre.csv', index=False)


predict_score()
gen_sub()
