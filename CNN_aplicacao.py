#!/usr/bin/env python
# coding: utf-8

## PACKAGE IMPORTATION ##

#TensorFlow e Keras
import tensorflow as tf
from tensorflow import keras

#Bibliotecas auxiliaresmuffin
import numpy as np
from matplotlib import pyplot as plt
import glob, os
import re
import pickle
import cv2

#Pillow
import PIL
from PIL import Image

########################################################################################################################################

## PREPROCESSING DATA ##

def jpeg_to_8_bit_greyscale(path, maxsize):

        img = Image.open(path)#.convert('L')   # convert image to 8-bit grayscale

        # Make aspect ratio as 1:1, by applying image crop.

    # Please note, croping works for this data set, but in general one

    # needs to locate the subject and then crop or scale accordingly.

        WIDTH, HEIGHT = img.size

        if WIDTH != HEIGHT:

                m_min_d = min(WIDTH, HEIGHT)

                img = img.crop((0, 0, m_min_d, m_min_d))

        # Scale the image to the requested maxsize by Anti-alias sampling.

        img.thumbnail(maxsize, PIL.Image.ANTIALIAS)

        return np.asarray(img)

def load_image_dataset(path_dir, maxsize):

        images = []

        labels = []

        os.chdir(path_dir)

        for file in glob.glob("*.jpg"):

                img = jpeg_to_8_bit_greyscale(file, maxsize)
                if re.match('*cataracts.jpg', file):

                        images.append(img)

                        labels.append(0)

                elif re.match('*dr.JPG', file):

                        images.append(img)

                        labels.append(1)

                elif re.match('*dmriDry.jpg', file):

                        images.append(img)

                        labels.append(2)


                elif re.match('*dmriWet.jpg', file):

                        images.append(img)

                        labels.append(3)


                elif re.match('*healthy.jpg', file):

                        images.append(img)

                        labels.append(4)
        return (np.asarray(images), np.asarray(labels))

maxsize = 512, 512

(train_images, train_labels) = load_image_dataset('datasets/dataset3/train', maxsize)

(test_images, test_labels) = load_image_dataset('datasets/dataset3/test', maxsize)

class_names = ['catarata', 'rd', 'dmriS', 'dmriM', 'saudavel']

print(train_images.shape)

print(train_labels)

print(test_images.shape)

IMG_SIZE = 512

data_train = [train_images, train_labels]

data_test = [test_images, test_labels]

print(len(data_train))

########################################################################################################################################

## GETTING DATA ##

import random

random.shuffle(data_train)


for exemplo in data_train:
    print(exemplo[1])


X_train = []
y_train = []


for features, label in data_train:
    X_train.append(features)
    y_train.append(label)
    
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("X.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close

picke_out = open("y.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close

for exemplo in data_train:
    print(exemplo[1])


X_test = []
y_test = []


for features, label in data_train:
    X_test.append(features)
    y_test.append(label)
    
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("X.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close

picke_out = open("y.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close

########################################################################################################################################

## CONVOLUTIONAL NERAL NETWORK ##

import tensorflow as tf

tf.enable_eager_execution()

def cria_rede(features, labels, mode):
    entrada = tf.reshape(features['X'], [-1, 512, 512, 3])
    #in [batch, 512, 512, 3]
    #out [batch, 256, 256, 96]
    convolucao1 = tf.layers.conv2d(inputs= entrada, filters= 32, kernel_size= [10,10], activation= tf.nn.relu,padding= 'same')
    pooling1 = tf.layers.max_pooling2d(inputs= convolucao1, pool_size= [2,2], strides= 2)
    
    #in [batch, 256, 256, 96]
    #out [batch, 128, 128, 96]
    convolucao2 = tf.layers.conv2d(inputs= pooling1, filters= 64, kernel_size= [10,10], activation= tf.nn.relu,padding= 'same')
    pooling2 = tf.layers.max_pooling2d(inputs= convolucao2, pool_size= [2,2], strides= 2)
    #in [batch, 128, 128, 96]
    #out [batch, 64, 64, 384]
    convolucao3 = tf.layers.conv2d(inputs= pooling2, filters= 128, kernel_size= [10,10], activation= tf.nn.relu,padding= 'same')
    pooling3 = tf.layers.max_pooling2d(inputs= convolucao3, pool_size= [2,2], strides= 2)

    flattening = tf.reshape(pooling3, [-1, 64*64*384])

    densa1 = tf.layers.dense(inputs= flattening, units= 1024, activation= tf.nn.relu)

    densa2 = tf.layers.dense(inputs= densa1, units= 1024, activation= tf.nn.relu)

    dropout = tf.layers.dropout(inputs= densa2, rate= 0.3, training = mode == tf.estimator.ModeKeys.TRAIN)

    saida = tf.layers.dense(inputs= dropout, units= 8)
    
    previsoes = tf.arg_max(saida, axis = 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = previsoes)
        
    erro = tf.looses.sparse_softmax_cross_entropy(label= labels, logits= saida)

    if mode == tf.estimator.ModeKeys.TRAIN:
        otimizador = tf.train.AdamOptmizer(learning_rate = 0.001)
        treinamento = otimizador.minimize(erro, global_step = tf.train.get_global_step)
        return tf.estimator.EstimatorSpec(mode= mode,loss= erro, train_op= treinamento)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {'acuracy': tf.metrics(lables = test_labels, predictions = previsoes)}
        return tf.estimator.EstimatorSpec(mode= mode, loss = erro, eval_metric_ops = eval_metrics_ops)

classificador = tf.estimator.Estimator(model_fn= cria_rede)


funcao_treinamento = tf.estimator.inputs.numpy_input_fn(x = {'X': X_train}, y = y_train, batch_size = 128, num_epochs = None, shuffle = True)
classificador.train(input_fn= funcao_treinamento, steps= 200)

funcao_teste = tf.estimator.inputs.numpy_input_fn(X = {'X': X_test, 'y': y_test}, num_epochs = 1, shuffle = False)

resultados = classificador.evaluate(input_fn= funcao_teste)
print('Resultados: ',resultados)

X_imagem_teste = X_test[0]
X_imagem_teste.shape()

X_imagem_teste = X_imagem_teste.reshapw(1, -1)
X_imagem_teste.shape()

funcao_previsao = tf.estimator.inputs.numpy_input_fn(X = {'X': X_imagem_teste}, shuffle = False)
pred= list(classificador.predict( input_fn = funcao_previsao))

plt.imshow(X_imagem_teste.reshape((512, 512)))
plt.title('Classe prevista: '+ str(pred[0]))

xgb_model = 0
# save the model XGBoost (xgb_model)
with open('CNN_aplicacao.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)