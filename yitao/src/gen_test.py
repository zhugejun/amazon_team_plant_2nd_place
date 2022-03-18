# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import  random
import  sys
import pickle

image_size = int(sys.argv[1])

print  " image_size = ", image_size
aug_type = 4
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
print  " gen_test   "

x_test = []

df_train = pd.read_csv('../input/train_v2.csv')
df_test = pd.read_csv('../input/sample_submission_v2.csv')


train_ids = np.array(df_train['image_name'])
test_ids = np.array(df_test['image_name'])

print  "train_ids.shape ", train_ids.shape

new_test_id_array = []

for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('../input/test-jpg/{}.jpg'.format(f))
    img = cv2.resize(img, (image_size, image_size))

    if aug_type == 4:
        x_test.append(img)
        new_test_id_array.append(f)

        x_test.append(randomRotate90(img, 90))
        new_test_id_array.append(f)

        x_test.append(randomRotate90(img, 180))
        new_test_id_array.append(f)

        x_test.append(randomRotate90(img, 270))
        new_test_id_array.append(f)

x_test = np.array(x_test, np.float16) / 255.
np.save('../data/X_test_aug' + str(aug_type) + '_' + str(image_size), x_test)

pickle.dump(new_test_id_array,open('../data/new_test_id_array' + str(aug_type) + '_' + str(image_size),'wb'))

