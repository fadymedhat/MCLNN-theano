import keras
import glob
import os
import numpy as np
from keras.models import Model
import matplotlib.cm as cm
import numpy as np
import numpy.ma as ma
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.set_printoptions(threshold=np.nan)

import keras
import glob
import os


class MinibatchPlotCallback(keras.callbacks.Callback):
    def nice_imshow(self, ax, data, vmin=None, vmax=None, cmap=None):
        """Wrapper around pl.imshow"""
        if cmap is None:
            cmap = cm.jet
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
        pl.colorbar(im, cax=cax)

    def make_mosaic(self, imgs, nrows, ncols, border=0):
        """
        Given a set of images with all the same shape, makes a
        mosaic with nrows and ncols
        """
        nimgs = imgs.shape[0]
        imshape = imgs.shape[1:]

        mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                                ncols * imshape[1] + (ncols - 1) * border),
                               dtype=np.float32)

        paddedh = imshape[0] + border
        paddedw = imshape[1] + border
        for i in xrange(nimgs):
            row = int(np.floor(i / ncols))
            col = i % ncols

            mosaic[row * paddedh:row * paddedh + imshape[0],
            col * paddedw:col * paddedw + imshape[1]] = imgs[i]
        return mosaic

    def __init__(self, filepath,X_train, X_test):
        self.learnedweightpath=filepath
        self.X_train = X_train
        self.X_test = X_test
    def on_epoch_end(self, epoch, logs={}):
        weightList = glob.glob(self.learnedweightpath + "*.hdf5")
        loccal_model=self.model


        modelIntermediate = Model(input=loccal_model.input,
                      output=loccal_model.get_layer('block1_prelu').output)  # block3_flatter block4_prelu block5_dropout
        block4_prelu = modelIntermediate.predict(self.X_train[0:1,:,:])
        block4_prelu1 = modelIntermediate.predict(self.X_train[130:131, :, :])
        #block4_prelu_test = modelIntermediate.predict(self.X_test)
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure(8)
        #h = modelIntermediate.get_weights()[0]
        #imgplot = plt.imshow(h[0, :, :])
        #imgplot = plt.imshow(np.transpose(block4_prelu[40000:40200, :]))
        b1 = np.squeeze(block4_prelu)
        b2 = np.squeeze(block4_prelu1)
        h = np.concatenate((b1, b2))
        #plt.imshow(np.transpose(block4_prelu[0, :, :]))
        plt.imshow(np.transpose(h))
        plt.pause(0.05)
        #plt.show()
        plt.ion()
        plt.figure(9)
        W = modelIntermediate.layers[2].W.get_value(borrow=True)
        W = np.squeeze(W)
        print("W shape : ", W)
        # #pl.figure()#figsize=(150, 150)
        # #pl.title('conv1 weights')
        # #self.nice_imshow(pl.gca(), self.make_mosaic(W, 6, 6), cmap=cm.binary)
        # rr = W.transpose(2, 1, 0)
        # self.nice_imshow(pl.gca(), self.make_mosaic(rr[0:30, :, :], 1, 30), cmap=cm.binary_r)
        # #t= modelIntermediate.get_weights()[0]
        # #imgplot = plt.imshow(t[0, :, :])
        # #plt.imshow(np.transpose(rr[0:30, :, :]))
        # plt.pause(0.05)





class DirectoryHouseKeepingCallback(keras.callbacks.Callback):
    def __init__(self, filepath):
        self.learnedweightpath=filepath
    def on_epoch_end(self, epoch, logs={}):
        weightList = glob.glob(os.path.join(self.learnedweightpath, "*.hdf5"))
        weightList.sort(key=os.path.getmtime)
        if len(weightList) > 60:
            os.remove(weightList[0])
        x= 6

