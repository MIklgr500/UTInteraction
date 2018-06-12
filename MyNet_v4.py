from keras import backend as K
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import ConvLSTM2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.engine.topology import get_source_inputs
#keras.io
from depthwise_conv2d import DepthwiseConvolution2D

def MobileNet(alpha=1.0, shape=[16,224,224,3], nframe=16):

    img_input = Input(shape)

    x = TimeDistributed(Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False))(img_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = ConvLSTM2D(int(128 * alpha), kernel_size=(7, 7), activation='relu', dropout=0.35, recurrent_dropout=0., return_sequences=False)(x)
    x = GlobalAveragePooling2D()(x)
    out=Dense(6, activation='softmax')(x)

    model = Model(img_input, out, name='mobilenet')
    return model
