import tflearn
import os
import numpy as np
import tensorflow as tf
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.data_utils import shuffle, to_categorical
from sklearn.cross_validation import train_test_split
from tflearn.layers.core import input_data, dropout, fully_connected

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(1e-3, '2conv-basic')

def conv_neural_network():

    tf.reset_default_graph()

    #network architecture
    input_layer = input_data(shape=[None, 50, 50, 1], name='input')

    convnet_layer = conv_2d(input_layer, 32, 5, activation='relu')
    convnet_layer = max_pool_2d(convnet_layer, 5)

    convnet_layer = conv_2d(convnet_layer, 64, 5, activation='relu')
    convnet_layer = max_pool_2d(convnet_layer, 5)

    convnet_layer = conv_2d(convnet_layer, 128, 5, activation='relu')
    convnet_layer = max_pool_2d(convnet_layer, 5)

    convnet_layer = conv_2d(convnet_layer, 64, 5, activation='relu')
    convnet_layer = max_pool_2d(convnet_layer, 5)

    convnet_layer = conv_2d(convnet_layer, 32, 5, activation='relu')
    convnet_layer = max_pool_2d(convnet_layer, 5)

    fc_layer = fully_connected(convnet_layer, 1024, activation='relu')
    fc_layer = dropout(fc_layer, 0.8)

    out_layer = fully_connected(fc_layer, 2, activation='softmax')
    out_layer = regression(out_layer, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(out_layer, tensorboard_dir='log')
    return model

def run(train_data, labels, model):
    # spliting training data into train_data and test_data
    train_x, test_x, train_y, test_y = train_test_split(train_data, labels, test_size=500, random_state=42)

    #reshaping x data
    train_x, test_x = train_x.reshape(-1, 50, 50, 1), test_x.reshape(-1, 50, 50, 1)

    #encoding y data
    train_y, test_y = to_categorical(train_y, 2), to_categorical(test_y, 2)

    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

    return model


##########################

