import sys
import pandas as pd
import os
import numpy as np

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
          'cloudy',
          'image_name']

dir_name = 'oof/'

mark = sys.argv[1]
print(mark)

cv_frame = pd.DataFrame()
test_frame = pd.DataFrame()

f2_scores = []
for filename in os.listdir(dir_name):
    if filename.startswith('oof_val_'):
        f2_score = float(filename.split('_')[-1].replace('.csv',''))
        f2_scores.append(f2_score)

        tmp = pd.read_csv(dir_name + filename)
        tmp = tmp[labels]
        cv_frame = pd.concat([cv_frame, tmp])
        print(cv_frame.shape)

    if filename.startswith('oof_test_prob_'):
        print(filename)

        tmp = pd.read_csv(dir_name + filename)
        tmp = tmp[labels]

        test_frame = pd.concat([test_frame, tmp])
        print(test_frame.shape)

cv_f2 = np.mean(f2_scores)
cv_frame = cv_frame.groupby(['image_name'])[labels].mean().reset_index()
cv_frame = cv_frame.sort_values(by='image_name', ascending=True)

test_frame = test_frame.groupby(['image_name'])[labels].mean().reset_index()
test_frame = test_frame.sort_values(by='image_name', ascending=True)

print(cv_f2)
cv_frame.to_csv('densenet121_' + str(mark) + '_cv.csv',index=False)
test_frame.to_csv('densenet121_' + str(mark) +'_fulltest.csv', index=False)

