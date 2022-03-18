#!/bin/bash


folder="../data"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi

python -u gen_cv.py 156
python -u gen_test.py 156

folder="../cv"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi

python -u train.py vgg19_156 156 VGG19

# find best threshold
folder="../thres"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi

python -u find_best_thre.py vgg19_156

# gen sub and cv
folder="../sub"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi
python -u predict_test.py vgg19_156 156 VGG19

