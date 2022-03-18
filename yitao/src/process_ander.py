
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

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

dir_name = '../d161/'

print  os.listdir(dir_name)

cv_frame = pd.DataFrame()
test_frame = pd.DataFrame()

for filename in os.listdir(dir_name):
    if filename.startswith('oof_val_'):
        print filename

        tmp = pd.read_csv(dir_name + filename)
        tmp = tmp[labels]
        cv_frame = pd.concat([cv_frame, tmp])
        print cv_frame.shape

    if filename.startswith('oof_test_prob_'):
        print filename

        tmp = pd.read_csv(dir_name + filename)
        tmp = tmp[labels]

        test_frame = pd.concat([test_frame, tmp])
        print test_frame.shape

print cv_frame.shape
cv_frame = cv_frame.groupby(['image_name'])[labels].mean().reset_index()
print cv_frame.shape

cv_frame = cv_frame.sort_values(by='image_name', ascending=True)

print test_frame.shape

test_frame = test_frame.groupby(['image_name'])[labels].mean().reset_index()

test_frame = test_frame.sort_values(by='image_name', ascending=True)
print test_frame.shape

cv_frame.to_csv('../d161/cv_'+'.csv',index=False)
print cv_frame.mean()
test_frame.to_csv('../d161/test_' +'.csv', index=False)


print test_frame.mean()
