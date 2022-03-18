#!/bin/bash


folder="../data"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi

python gen_cv.py 128
python gen_cv.py 156
python gen_cv.py 196
python gen_cv.py 208
python gen_cv.py 256


python gen_test.py 128
python gen_test.py 156
python gen_test.py 196
python gen_test.py 208
python gen_test.py 256


#  train
folder="../cv"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi

python -u train.py vgg19_156 156 VGG19
python -u train.py res50_208 208 ResNet50
python -u train.py v3_156 156 InceptionV3
python -u train.py dense_128 128 DenseNet
python -u train.py dense_256 256 DenseNet
python -u train.py x_128 128 Xception
python -u train.py x_256 256 Xception
python -u train.py vgg19_256 256 VGG19
python -u train.py vgg16_196 196 VGG16


# find best threshold
folder="../thres"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi

python -u find_best_t.py vgg19_156
python -u find_best_t.py res50_208
python -u find_best_t.py v3_156
python -u find_best_t.py dense_128
python -u find_best_t.py dense_256
python -u find_best_t.py x_128
python -u find_best_t.py x_256
python -u find_best_t.py vgg19_256
python -u find_best_t.py vgg16_196

# find thres for gejun's model
python -u find_best_t.py densenet121_1
python -u find_best_t.py densenet121_2


# predict

folder="../sub"
if [ ! -d "$folder" ] ; then
  mkdir "$folder"
fi


python -u predict_test.py vgg19_156 156 VGG19
python -u predict_test.py res50_208 208 ResNet50
python -u predict_test.py v3_156 156 InceptionV3
python -u predict_test.py dense_128 128 DenseNet
python -u predict_test.py dense_256 256 DenseNet
python -u predict_test.py x_128 128 Xception
python -u predict_test.py x_256 256 Xception
python -u predict_test.py vgg19_256 256 VGG19
python -u predict_test.py vgg16_196 196 VGG16

python -u avg_all.py