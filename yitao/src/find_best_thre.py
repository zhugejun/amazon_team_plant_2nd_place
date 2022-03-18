# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import sys
import pickle

warnings.filterwarnings('ignore')

from sklearn.metrics import fbeta_score

labels = ['blow_down', 'bare_ground', 'conventional_mine', 'blooming', 'cultivation', 'artisinal_mine',
          'haze', 'primary', 'slash_burn', 'habitation', 'clear', 'road', 'selective_logging', 'partly_cloudy',
          'agriculture', 'water', 'cloudy']

def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=2, average='samples')

def get_optimal_threshhold(true_label, prediction, iterations=200):
    best_threshhold = [0.2]*17

    temp_fbeta = fbeta(true_label, prediction > best_threshhold)
    print "temp_fbeta： ", temp_fbeta, "   best_thre ", best_threshhold

    for t in tqdm(range(17)):
        temp_threshhold = np.array([0.2]*17)
        best_fbeta = 0
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if  temp_fbeta >= best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value

    print best_threshhold

    print "best score  ", fbeta(true_label, prediction > best_threshhold)

    return best_threshhold



print  " \n\n ****************   "
print  " thresholding    "
tag = sys.argv[1]

df = pd.read_csv('../input/train_v2.csv')
cv_score = pd.read_csv('../cv/' + tag + '_cv.csv')

print cv_score.shape

cv_score = cv_score.groupby(['image_name'])[labels].mean().reset_index()
print cv_score.shape

com_frame = pd.merge(df, cv_score, how='inner', on='image_name')

print com_frame.shape


cols = list(com_frame.columns)
print cols
del cols[0]
del cols[0]
print cols

label_map = {l: i for i, l in enumerate(cols)}
print label_map
inv_label_map = {i: l for l, i in label_map.items()}
print inv_label_map
y_true = []
y_pred = []


for index, row in com_frame.iterrows():
    targets = np.zeros(17)
    p = []
    for t in row['tags'].split(' '):
        targets[label_map[t]] = 1
    y_true.append(targets)
    y_pred.append(row[cols])

print len(y_pred)

best_threshold = get_optimal_threshhold(np.array(y_true), np.array(y_pred) )
pickle.dump(best_threshold,open('../thres/thre_' + str(tag),'wb'))
