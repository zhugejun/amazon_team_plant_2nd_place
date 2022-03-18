#!/bin/bash
echo '***************** Creating n folder *****************'

python create_n_folder.py

echo '***************** Training without flip *****************'
echo '-------------------Folder 0----------------------'
python train_n_fold_no_flip.py 0

echo '-------------------Folder 1----------------------'
python train_n_fold_no_flip.py 1

echo '-------------------Folder 2----------------------'
python train_n_fold_no_flip.py 2

echo '-------------------Folder 3----------------------'
python train_n_fold_no_flip.py 3

echo '-------------------Folder 4----------------------'
python train_n_fold_no_flip.py 4


echo '***************** Predicting *****************'
echo '-------------------Folder 0----------------------'
python multiple_preds_for_test.py 0

echo '-------------------Folder 1----------------------'
python multiple_preds_for_test.py 1

echo '-------------------Folder 2----------------------'
python multiple_preds_for_test.py 2

echo '-------------------Folder 3----------------------'
python multiple_preds_for_test.py 3

echo '-------------------Folder 4----------------------'
python multiple_preds_for_test.py 4

echo '***************** Combine  *****************'
python process_ander.py 1

rm oof/oof_*.csv

echo '***************** Training *****************'
echo '-------------------Folder 0----------------------'
python train_n_fold_w_flip.py 0

echo '-------------------Folder 1----------------------'
python train_n_fold_w_flip.py 1

echo '-------------------Folder 2----------------------'
python train_n_fold_w_flip.py 2

echo '-------------------Folder 3----------------------'
python train_n_fold_w_flip.py 3

echo '-------------------Folder 4----------------------'
python train_n_fold_w_flip.py 4


echo '***************** Predicting *****************'
echo '-------------------Folder 0----------------------'
python multiple_preds_for_test.py 0

echo '-------------------Folder 1----------------------'
python multiple_preds_for_test.py 1

echo '-------------------Folder 2----------------------'
python multiple_preds_for_test.py 2

echo '-------------------Folder 3----------------------'
python multiple_preds_for_test.py 3

echo '-------------------Folder 4----------------------'
python multiple_preds_for_test.py 4

echo '***************** Combine  *****************'
python process_ander.py 2
