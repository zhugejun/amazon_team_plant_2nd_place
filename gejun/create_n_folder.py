import pandas as pd
from sklearn.model_selection import KFold
import os
import settings

NFOLDS = 5

train = pd.read_csv(settings.DATA_DIR + 'train_v2.csv')

def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res

rotate_angle = pd.DataFrame({'angle': [0, 90, 180, 270]})

train_new = df_crossjoin(train, rotate_angle)
train_new['id'] = [i for i in range(len(train) * 4)]
train_new = train_new.set_index(['id'])

kf = KFold(n_splits = NFOLDS, shuffle=True, random_state=2016920)

for num_folder, (tra_idx, val_idx) in enumerate(kf.split(train_new)):
    print(num_folder)

    X_tra = pd.DataFrame()
    X_tra['image_name'] = train_new['image_name'][tra_idx]
    X_tra['tags'] = train_new['tags'][tra_idx]
    X_tra['angle'] = train_new['angle'][tra_idx]

    X_val = pd.DataFrame()
    X_val['image_name'] = train_new['image_name'][val_idx]
    X_val['tags'] = train_new['tags'][val_idx]
    X_val['angle'] = train_new['angle'][val_idx]


    X_tra.to_csv('oof/tra_data_' + str(num_folder) + '.csv', index=False)
    X_val.to_csv('oof/val_data_' + str(num_folder) + '.csv', index=False)
