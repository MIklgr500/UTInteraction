import cv2
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.transform import resize as imgresize
from scipy import linalg
import scipy.ndimage as ndi
from scipy.signal import convolve2d
import random
from keras.utils import to_categorical
from tqdm import tqdm

# keras.io
def rotation(x, rg, row_axis=0, col_axis=1, channel_axis=2,
               fill_mode='nearest', cval=0.):
    """ Like in keras lib, but
            rg: random angel
    """
    theta = np.deg2rad(rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[int(row_axis)], x.shape[int(col_axis)]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def shift(x, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2,
                 fill_mode='nearest', cval=0.):
    """ Like in keras lib, but
            wrg, hrg: random number
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hrg * h
    ty = wrg * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def zoom(x, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: ', zoom_range)

    zx, zy = zoom_range
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=2,
                    fill_mode='nearest',
                    cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis+1)
    return x

def tranformation(x, config):
    if config['flag']:
        if config['flip']:
            x = x[:, ::-1, :]
        if config['rot']['flag']:
            x = rotation(x, config['rot']['rg'])
        if config['shift']['flag']:
            x = rotation(x, config['shift']['wrg'], config['shift']['hrg'])
        if config['zoom']['flag']:
            x = zoom(x, config['zoom']['zoom_range'])
    return x

def msqr(x1,x2):
    return np.mean(np.sqrt(np.power(x1-x2,2)))

def new_frame(x1,x2, op, kernel_size=4):
    w, h, _ = np.shape(x1)
    new_frame = np.zeros(shape=(w//kernel_size, h//kernel_size, 1))

    if op == 'msqr':
        x1 = np.asarray(x1)/255
        x2 = np.asarray(x2)/255
        func = lambda n1, n2: msqr(n1, n2)
    else:
        raise "Undefined fuction: {0}".format(op)

    for i in range(0, w//kernel_size):
        for j in range(0, h//kernel_size):
            sx1 = x1[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size,:]
            sx2 = x2[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size,:]
            new_frame[i,j,0] = func(sx1,sx2)
    return new_frame

class DataSet:
    def __init__(self,
                 nframe=16,
                 fstride=1,
                 train_size=0.7,
                 size=[224,224, 3],
                 random_state=131,
                 name = "",
                 filepaths=[],
                 y=[],
                 kernel_size=4):
        y = to_categorical(y=y, num_classes=6)
        train_fp, valid_fp, train_y, valid_y = train_test_split(filepaths,
                                                                y,
                                                                random_state=random_state,
                                                                shuffle=True,
                                                                train_size=train_size,
                                                                stratify=y
                                                )
        self.train_fp = train_fp
        self.valid_fp = valid_fp
        self.train_y = train_y
        self.valid_y = valid_y
        self.nframe = nframe
        self.fstride = fstride
        self.size = size
        self.random_state = random_state
        self.train_size = train_size
        self.name = name
        self.kernel_size = kernel_size

    def _augmentation(self, img, config={}):
        img = rgb2gray(img)
        img = img.reshape([self.size[0]//self.kernel_size, self.size[1]//self.kernel_size, 1])
        img = tranformation(img, config)
        return img

    def _reader(self, filepath, aug_config={'flag':False}):
        if aug_config['flag']:
            rg = random.uniform(-aug_config['rg'], aug_config['rg'])
            wrg = random.uniform(-aug_config['wrg'], aug_config['wrg'])
            hrg = random.uniform(-aug_config['hrg'], aug_config['hrg'])
            zoom_range = np.random.uniform(1-aug_config['zoom'], 1+aug_config['zoom'], 2)
            fr = bool(0.5<random.uniform(0, 1))
            fs = bool(0.5<random.uniform(0, 1))
            fz = bool(0.5<random.uniform(0, 1))
            ff = bool(0.5<random.uniform(0, 1))

            config = {
                'flag':True,
                'rot':{
                    'flag':fr,
                    'rg': rg
                },
                'shift':{
                    'flag':fs,
                    'wrg': wrg,
                    'hrg': hrg
                },
                'zoom':{
                    'flag':fz,
                    'zoom_range':zoom_range
                },
                'flip': ff
            }
        else:
            config = {
                'flag':False
            }
        video = []
        cap = cv2.VideoCapture(filepath)
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret==True:
                img = self._augmentation(img, config=config)
                video.append(img)
            else:
                break
        cap.release()
        return np.array(video)

    def _vis_video(self, video, y, name='standartName.avi'):
        out = cv2.VideoWriter(name,
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              10,
                              (self.size[0]//self.kernel_size, self.size[1]//self.kernel_size),
                              False)
        for frame in video:
            frame = frame*255
            frame = frame.astype(np.uint8)
            print(y)
            #cv2.putText(frame, str(y),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
            out.write(frame.reshape((self.size[0]//self.kernel_size, self.size[1]//self.kernel_size)))
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
        out.release()
        cv2.destroyAllWindows()

    def visualizer(self, num=10, type='train', aug_config={'flag':False}):
        if type=='train':
            gen = self.train_gen(batch_size=num, aug_config=aug_config)
        else:
            gen = self.valid_gen(batch_size=num)
        x, y = next(gen)
        for i in range(num):
            self._vis_video(video=x[i], y=y[i].argmax(-1), name=str(i)+'_'+str(y[i].argmax(-1))+'.avi')


    def make_set(self, op='msqr', name='train'):
        if name=='train':
            path = self.train_fp
        elif name=='valid':
            path = self.valid_fp
        else:
            raise 'name must be train or valid'
        for i, fp in tqdm(enumerate(path),ascii=True, desc='Make {0} Set'.format(name)):
            out = cv2.VideoWriter(name+'_set/'+str(i)+'.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'),
                                  5,
                                  (self.size[0]//self.kernel_size, self.size[1]//self.kernel_size),
                                  False)
            i,j=0,1
            video = []
            cap = cv2.VideoCapture(fp)
            pr_img = np.zeros(shape=(self.size[0]//self.kernel_size, self.size[1]//self.kernel_size,1))
            last_img = np.zeros(shape=(self.size[0]//self.kernel_size, self.size[1]//self.kernel_size))
            while(cap.isOpened()):
                ret, img = cap.read()
                if ret==True:
                    if bool(i%self.fstride):
                        i+=1
                        continue
                    if j==(self.nframe):
                        break
                    img = imgresize(img, self.size)
                    img = rgb2gray(img)
                    img = img.reshape([self.size[0], self.size[1], 1])
                    if i==0:
                        img1 = new_frame(img,img, op=op, kernel_size=self.kernel_size)
                        pr_img = np.copy(img)
                        video.append(img1.reshape((self.size[0]//self.kernel_size, self.size[1]//self.kernel_size)))
                        i+=1
                        j+=1
                    else:
                        img1 = new_frame(img,pr_img, op=op, kernel_size=self.kernel_size)
                        pr_img = np.copy(img)
                        last_img = np.copy(img1.reshape((self.size[0]//self.kernel_size, self.size[1]//self.kernel_size)))
                        video.append(img1.reshape((self.size[0]//self.kernel_size, self.size[1]//self.kernel_size)))
                        i+=1
                        j+=1
                else:
                    break

            while j!=self.nframe:
                video.append(last_img)
                j+=1
            video = np.array(video)/np.max(video)
            for f in video:
                f = f*255
                f = f.astype(np.uint8)
                out.write(f)
            out.release()
            cap.release()

    def train_gen(self, batch_size=1, aug_config={'flag':False}):
        iarr = [i for i in range(len(self.train_y))]
        while True:
            bn = np.random.choice(iarr, size=batch_size)
            v_batch = []
            y_batch = []
            for i in range(batch_size):
                fp = 'train_set/'+str(bn[i])+'.avi'
                y = self.train_y[bn[i]]
                v = self._reader(fp, aug_config)
                v_batch.append(v/255)
                y_batch.append(np.array(y))
            yield [np.array(v_batch), np.array(y_batch)]

    def valid_gen(self):
        while True:
            for bn in range(len(self.valid_y)):
                v_batch = []
                y_batch = []
                fp = 'valid_set/'+str(bn)+'.avi'
                y = self.valid_y[bn]
                v = self._reader(fp)
                v_batch.append(v/255)
                y_batch.append(np.array(y))
                yield [np.array(v_batch), np.array(y_batch)]
    @property
    def getVlen(self):
        return len(self.valid_y)

    def __str__(self):
        return "DataSet "+str(self.name)+"\n"+\
                "N train samples: "+str(len(self.train_y))+"\n"+\
                "N valid samples: "+str(len(self.valid_y))+"\n"+\
                "Frame Stride: "+str(self.fstride)+"\n"+\
                "N Frame: "+str(self.nframe)+"\n"+\
                "Size: "+str(self.size)+"\n"+\
                "Random State: "+str(self.random_state)+"\n"
