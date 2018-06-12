from keras import backend as K
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from depthwise_conv2d import DepthwiseConvolution2D


def MyNet(alpha=1.0, shape=[16,224,224,3], nframe=16):

    img_input = Input(shape)

    x = ConvLSTM2D(32, kernel_size=(3,3), activation='relu')(img_input)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.)(x)
    out=Dense(6, activation='softmax')(x)

    model = Model(img_input, out, name='mobilenet')
    return model
