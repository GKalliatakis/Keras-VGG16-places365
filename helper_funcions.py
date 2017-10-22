from keras.layers import Merge
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
import numpy as np


def transform_conv_weight(W):
    ''' For non FC layers, do this because Keras does convolution vs Caffe's correlation '''
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)

    W = np.transpose(W)

    return W


def transform_fc_weight(W):
    '''The weights for fully-connected layers are transposed between Caffe and Keras'''
    return W.T