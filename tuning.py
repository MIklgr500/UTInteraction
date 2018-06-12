from reader import DataSet
from mobilenet import MobileNet
from my_net import MyNet
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from keras import backend as K
from keras import metrics
from keras import losses
from keras.utils import to_categorical
import os
import numpy as np

def search_file(init_path, paths=[], y=[]):
    for root, dirs, files in os.walk(init_path):
        for f in files:
            paths.append(os.path.join(init_path,f))
            y.append(int(f[-5]))
    return paths, y

def get_callbacks(filepath, patience=1):
    mcp = ModelCheckpoint(filepath+'.h5',
                          monitor='val_acc',
                          verbose=1,
                          save_best_only=True,
                          save_weights_only=False,
                          mode='max',
                          period=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1,
                              patience=patience, min_lr=1e-6)
    csv_log = CSVLogger(filename=filepath+'.csv')
    return [mcp, rlr, csv_log]

def train():
    tr_config = {
        'flag':True,
        'rg':25, # 7, 5
        'wrg':0.25, # 1, 3
        'hrg':0.25, # 1, 3
        'zoom':0.25 # 1, 1
    }
    callbacks = get_callbacks('mynet_v4_bias', patience=30)

    paths, y = search_file('set1/segmented_set1')
    paths, y = search_file('set2/segmented_set2', paths=paths, y=y)

    ds = DataSet(nframe=30, fstride=6, name='UT interaction', size=[224, 224, 3], filepaths=paths, y=y, kernel_size=4)
    ds.make_set(op='msqr',name='train')
    ds.make_set(op='msqr', name='valid')

    #opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.1)
    #opt = SGD(lr=2*1e-1, momentum=0.9, nesterov=True, decay=0.2)
    opt = RMSprop(lr=0.001, rho=0.9, decay=0.01)

    model = MobileNet(alpha=1.0, shape=[29,56,56,1], nframe=29)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    #model.load_weights('mynet_v4.h5')
    model.fit_generator(generator=ds.train_gen(batch_size=5,aug_config=tr_config),
                        steps_per_epoch=100,
                        epochs=300,
                        validation_data=ds.valid_gen(),
                        verbose=1,
                        validation_steps=ds.getVlen,
                        callbacks=callbacks)
