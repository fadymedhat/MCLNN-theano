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
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

import keras
import glob
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras import callbacks
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as pyplt
import matplotlib
import matplotlib.pyplot as plt
from visualization import Visualizer


class SegmentPlotCallback(keras.callbacks.Callback):
    def __init__(self, configuration, data_loader):
        self.configuration = configuration
        self.visualization_parent_path = configuration.VISUALIZATION_PARENT_PATH
        self.visualizer = Visualizer(configuration)
        self.data_loader = data_loader

        self.train_path = os.path.join(self.visualization_parent_path, 'train_sample_across_epochs')
        self.test_path = os.path.join(self.visualization_parent_path, 'test_sample_across_epochs')
        self.validation_path = os.path.join(self.visualization_parent_path, 'validation_sample_across_epochs')

        # change the following numbers to choose other segments for visualization
        self.train_segment_index = configuration.TRAIN_SEGMENT_INDEX
        self.test_segment_index = configuration.TEST_SEGMENT_INDEX
        self.validation_segment_index = configuration.VALIDATION_SEGMENT_INDEX
        #
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        if not os.path.exists(self.validation_path):
            os.makedirs(self.validation_path)

    def on_epoch_end(self, epoch, logs={}):


        self.visualizer.visualize_prediction_segments(self.model,
                                                      np.expand_dims(
                                                          self.data_loader.train_segments[self.train_segment_index], 0),
                                                      path=self.train_path,
                                                      initial_segment=self.train_segment_index,
                                                      epoch_id='epoch_'+str(epoch),
                                                      layer_filter_list=['mclnn'], first_mclnn_only=True)

        self.visualizer.visualize_prediction_segments(self.model,
                                                      np.expand_dims(
                                                          self.data_loader.test_segments[self.test_segment_index], 0),
                                                      path=self.test_path,
                                                      initial_segment=self.test_segment_index,
                                                      epoch_id='epoch_'+str(epoch),
                                                      layer_filter_list=['mclnn'], first_mclnn_only=True)

        self.visualizer.visualize_prediction_segments(self.model,
                                                      np.expand_dims(self.data_loader.validation_segments[
                                                                         self.validation_segment_index], 0),
                                                      path=self.validation_path,
                                                      initial_segment=self.validation_segment_index,
                                                      epoch_id='epoch_'+str(epoch),
                                                      layer_filter_list=['mclnn'], first_mclnn_only=True)


class DirectoryHouseKeepingCallback(keras.callbacks.Callback):
    def __init__(self, filepath):
        self.learnedweightpath = filepath

    def on_epoch_end(self, epoch, logs={}):
        weightList = glob.glob(os.path.join(self.learnedweightpath, "*.hdf5"))
        weightList.sort(key=os.path.getmtime)
        if len(weightList) > 60:
            os.remove(weightList[0])


def prepare_callbacks(configuration, fold_weights_path, data_loader):
    callback_list = []

    # remote_callback = callbacks.RemoteMonitor(root='http://localhost:9000')
    # callback_list.append(remote_callback)

    # early stopping
    early_stopping_callback = callbacks.EarlyStopping(monitor=configuration.STOPPING_CRITERION,
                                                      patience=configuration.WAIT_COUNT,
                                                      verbose=0,
                                                      mode='auto')
    callback_list.append(early_stopping_callback)

    # save weights at the end of epoch
    weights_file_name_format = 'weights.epoch{epoch:02d}-val_loss{val_loss:.2f}-val_acc{val_acc:.4f}.hdf5'
    checkpoint_callback = ModelCheckpoint(os.path.join(fold_weights_path, weights_file_name_format),
                                          monitor='val_loss', verbose=0,
                                          save_best_only=False, mode='auto')
    callback_list.append(checkpoint_callback)

    # free space of stored weights of early epochs
    directory_house_keeping_callback = DirectoryHouseKeepingCallback(fold_weights_path)
    callback_list.append(directory_house_keeping_callback)

    # call for visualization if it is enabled
    if configuration.SAVE_SEGMENT_PREDICTION_IMAGE_PER_EPOCH == True:
        segment_plot_callback = SegmentPlotCallback(configuration=configuration,
                                                    data_loader=data_loader)
        callback_list.append(segment_plot_callback)

    return callback_list
