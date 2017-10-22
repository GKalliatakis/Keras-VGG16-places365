'''VGG16-places365 pre-trained model for Keras

# Reference:
- [Places: A 10 million Image Database for Scene Recognition](http://places2.csail.mit.edu/PAMI_places.pdf)
'''

from __future__ import division, print_function
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import numpy as np
from keras.layers import Conv2D
from keras.regularizers import l2
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from helper_funcions import transform_conv_weight,transform_fc_weight

from keras.applications.imagenet_utils import _obtain_input_shape

import os

CAFFE_WEIGHTS_DIR = "/path/to/downloaded/weights



def VGG16_Places365(include_top=True, weights='places',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=365):
    """Instantiates the VGG16-places365 architecture.

    Optionally loads weights pre-trained
    on Places. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "places" (pre-training on Places).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape
        """

    if weights not in {'places', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `places` '
                         '(pre-training on Places).')

    if weights == 'places' and include_top and classes != 365:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 365')


    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    W_conv1_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1_1.npy")))
    W_conv1_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1_2.npy")))

    b_conv1_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1_1.npy"))
    b_conv1_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1_2.npy"))

    # Block 2
    W_conv2_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2_1.npy")))
    W_conv2_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2_2.npy")))

    b_conv2_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2_1.npy"))
    b_conv2_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2_2.npy"))

    # Block 3
    W_conv3_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3_1.npy")))
    W_conv3_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3_2.npy")))
    W_conv3_3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3_3.npy")))

    b_conv3_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3_1.npy"))
    b_conv3_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3_2.npy"))
    b_conv3_3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3_3.npy"))

    # Block 4
    W_conv4_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4_1.npy")))
    W_conv4_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4_2.npy")))
    W_conv4_3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4_3.npy")))

    b_conv4_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4_1.npy"))
    b_conv4_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4_2.npy"))
    b_conv4_3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4_3.npy"))

    # Block 5
    W_conv5_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv5_1.npy")))
    W_conv5_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv5_2.npy")))
    W_conv5_3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv5_3.npy")))

    b_conv5_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv5_1.npy"))
    b_conv5_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv5_2.npy"))
    b_conv5_3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv5_3.npy"))

    W_fc6 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc6.npy")))
    b_fc6 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc6.npy"))

    W_fc7 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc7.npy")))
    b_fc7 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc7.npy"))

    W_fc8a = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc8a.npy")))
    b_fc8a = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc8a.npy"))

    # Because the dimension ordering of the layer outputs and the weights of the
    # Caffe network are all in Theano (channel, row, col) format.
    # K.set_image_dim_ordering("th")

    K.set_image_data_format('channels_first')

    data = Input(shape=(3, 224, 224), name="data")

    # Block 1
    conv1_1 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv1_1, b_conv1_1),
                     activation='relu', name='conv1_1')(data)

    conv1_2 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv1_2, b_conv1_2),
                     activation='relu', name='conv1_2')(conv1_1)

    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1", padding='valid')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv2_1, b_conv2_1),
                     activation='relu', name='conv2_1')(pool1)

    conv2_2 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv2_2, b_conv2_2),
                     activation='relu', name='conv2_2')(conv2_1)

    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2", padding='valid')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv3_1, b_conv3_1),
                     activation='relu', name='conv3_1')(pool2)

    conv3_2 = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv3_2, b_conv3_2),
                     activation='relu', name='conv3_2')(conv3_1)

    conv3_3 = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv3_3, b_conv3_3),
                     activation='relu', name='conv3_3')(conv3_2)

    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool3", padding='valid')(conv3_3)

    # Block 4
    conv4_1 = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv4_1, b_conv4_1),
                     activation='relu', name='conv4_1')(pool3)

    conv4_2 = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv4_2, b_conv4_2),
                     activation='relu', name='conv4_2')(conv4_1)

    conv4_3 = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv4_3, b_conv4_3),
                     activation='relu', name='conv4_3')(conv4_2)

    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool4", padding='valid')(conv4_3)

    # Block 5
    conv5_1 = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv5_1, b_conv5_1),
                     activation='relu', name='conv5_1')(pool4)

    conv5_2 = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv5_2, b_conv5_2),
                     activation='relu', name='conv5_2')(conv5_1)

    conv5_3 = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
                     kernel_regularizer=l2(0.0002), weights=(W_conv5_3, b_conv5_3),
                     activation='relu', name='conv5_3')(conv5_2)

    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool5", padding='valid')(conv5_3)

    if include_top:
        # Classification block
        fc6 = Flatten(name='flatten')(pool5)
        fc6 = Dense(4096, activation='relu', name='fc6', weights=(W_fc6, b_fc6), )(fc6)
        drop6 = Dropout(0.5, name='drop6')(fc6)

        fc7 = Dense(4096, activation='relu', name='fc7', weights=(W_fc7, b_fc7), )(drop6)
        drop7 = Dropout(0.5, name='drop7')(fc7)

        fc8a = Dense(365, name="fc8a", weights=(W_fc8a, b_fc8a), )(drop7)

        prob = Activation("softmax", name="prob")(fc8a)

    else:
        if pooling == 'avg':
            prob = GlobalAveragePooling2D()(pool5)
        elif pooling == 'max':
            prob = GlobalMaxPooling2D()(pool5)



    # Create model.
    model = Model(data, prob, name='vgg16_places_365')

    return model


if __name__ == '__main__':
    model = VGG16_Places365(include_top=True, weights='places')

