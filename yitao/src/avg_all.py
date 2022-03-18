
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tqdm

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

ids = []
full_array = []

def read_file(file_name, idx):
    global full_array
    global ids
    print  'reading....', file_name
    test_frame = pd.read_csv(file_name)
    test_frame = test_frame.groupby(['image_name'])[labels].mean().reset_index()

    test_frame = test_frame.sort_values(by='image_name', ascending=True)
    ids = test_frame['image_name']
    tmp = test_frame[labels]
    tmp_array = np.array(tmp)
    print tmp_array.shape
    if idx == 0:
        full_array = tmp_array
    else:
        full_array += tmp_array



tags = ['vgg19_156','res50_208','v3_156','dense_128','dense_256','x_128'
     ,'x_256','vgg19_256','vgg16_196', 'densenet121_1', 'densenet121_2']

#tags = ['vgg19_156','res50_208','vgg16_196']

for i in range(0, len(tags)):
    filename = '../cv/' + tags[i] + '_fulltest.csv'
    if os.path.exists(filename):
        read_file(filename,i)
    else:
        print filename, '  not exist!'

full_array = full_array / len(tags)
print full_array.shape

result = pd.DataFrame(full_array, columns=labels)

print result.head(2)
print result.mean()

thres = []

for i in range(0, len(tags)):
    filename = '../thres/thre_' + str(tags[i])
    if os.path.exists(filename) == False:
        print filename, ' not exist!'
        continue
    print 'load... ', filename
    tmp = np.load(filename)
    tmp = np.array(tmp)
    if i == 0:
        thres = tmp
    else:
        thres += tmp
thres = thres/len(tags)

print thres
sub_frame = pd.DataFrame()
sub_frame['image_name'] = ids
preds = []
for i in range(result.shape[0]):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))
sub_frame['tags'] = preds
sub_frame.to_csv('../sub/ensemble11.csv', index=False)