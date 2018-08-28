"""
The data processor loads the data, labels and indices.
It applies the necessary pre-processing for the MCLNN
"""
# from __future__ import division
import os
import h5py
import numpy as np
import scipy.io as sio
from keras.utils import np_utils


class DataLoader(object):
    train_segments = None
    train_labels = None

    test_segments = None
    test_labels = None

    validation_segments = None
    validation_labels = None

    test_clips = None
    test_clips_labels = None

    def load_fold_with_labels(self, fold_name, data_path, index_path):
        """

        :param fold_name:
        :param data_path:
        :param index_path:
        :return:
        """
        print('Loading ' + fold_name + ' fold ...')
        with h5py.File(index_path, "r") as hdf5_handle:
            index = hdf5_handle[str('index')].value #hdf5_handle[str('index')].value  #  np.asarray([0, 1]) #
            label = hdf5_handle[str('label')].value #hdf5_handle[str('label')].value  # np.asarray( [0, 1])

        sound_cell_array = []
        category = []
        with h5py.File(data_path, 'r') as f:
            # print('List of arrays in this file: \n', f.keys())
            for i in range(len(index)): #range(len(index)):    # range(2): to execlude the category list dataset vector
                data = f[str(index[i])].value
                sound_cell_array.append(data)

        samples_count = len(sound_cell_array)
        clip_frame_count = len(sound_cell_array[0])
        features_count = len(sound_cell_array[0][0])

        print(fold_name + ' - path =\'' + data_path + '\'')
        print(fold_name + ' - index path =\'' + index_path + '\'')
        print(fold_name + ' - samples count =\'' + str(samples_count)
              + '\' - clip frame count =\'' + (str(
            clip_frame_count))  # if len(sound_cell_array.shape)> 1 else 'variable length') #str(sound_cell_array.shape[1])
              + '\' - features count =\'' + (
                  str(features_count))  # if len(sound_cell_array.shape)> 1 else str(sound_cell_array[0].shape[1]))
              + '\' - indices count =\'' + str(index.shape[0])
              + '\' - labels count =\'' + str(label.shape[0]) + '\'')
        return sound_cell_array, label

    def retrieve_standardization_parameters(self, data, train_index_path):
        """

        :param data:
        :return:
        """

        samples = str(len(data))
        frames = str(len(data[0]))
        features = str(len(data[0][0]))

        standardization_file_path = train_index_path.replace('.hdf5', 'Parameters.hdf5').replace('_index',
                                                                                                 '_standardization')

        if not os.path.exists(standardization_file_path):
            print ('Calculating standardization parameters ...')

            mean_vector = np.mean(np.concatenate(data, axis=0), axis=0)
            std_vector = np.std(np.concatenate(data, axis=0), axis=0)

            if not os.path.exists(os.path.dirname(standardization_file_path)):
                os.makedirs(os.path.dirname(standardization_file_path))

            hdf5_handle = h5py.File(standardization_file_path, "w")

            hdf5_handle.create_dataset(str('mean_vector'), data=mean_vector,
                                       dtype='float32')
            hdf5_handle.create_dataset(str('std_vector'), data=std_vector,
                                       dtype='float32')

        else:
            print ('Loading standardization parameters ...')
            with h5py.File(standardization_file_path, "r") as hdf5_handle:
                mean_vector = hdf5_handle[str('mean_vector')].value
                std_vector = hdf5_handle[str('std_vector')].value

        print(standardization_file_path)
        print(
            'Train data size in 3D [samples, frames, feature vector] = (' + samples + ', ' + frames + ', ' + features + ')')

        return mean_vector, std_vector

    def standardize_data(self, data, mean_vector, std_vector):
        """

        :param data:
        :param mean_vector:
        :param std_vector:
        :return:
        """
        sample_count = len(data)
        for i in range(sample_count):
            data[i] = (data[i] - mean_vector) / std_vector
        return data

    def segment_fold(self, fold_name, data, label, segment_size, step_size):
        """

        :param fold_name:
        :param data:
        :param label:
        :param segment_size:
        :param step_size:
        :return:
        """

        data_segments = []
        segments_labels = []

        sample_count = len(data)
        for i in range(0, sample_count):
            for j in range(0, data[i].shape[0] - segment_size, step_size):
                data_segments.append(data[i][j:(j + segment_size + 1), :])
                segments_labels.append(np.asarray(label[i]))

        data_segments = np.asarray(data_segments)
        segments_labels = np.asarray(segments_labels)

        print (fold_name + '- samples ' + str(sample_count) + ' - segments ' + str(
            data_segments.shape[0]) + ' - Step Size is: ' +
               str(step_size) + '  i.e. Segments overlap with q-' +
               str(step_size) + ' frames')

        return data_segments, segments_labels

    def segment_clip(self, data, label, segment_size, step_size):
        """

        :param data:
        :param label:
        :param segment_size:
        :param step_size:
        :return:
        """
        data_segments = []
        segments_labels = []
        for j in range(0, data.shape[0] - segment_size, step_size):
            data_segments.append(data[j:(j + segment_size + 1), :])
            segments_labels.append(label)
        return np.asarray(data_segments), np.asarray(segments_labels)

    def load_data(self, segment_size, step_size, nb_classes, data_path, train_index_path, test_index_path,
                  validation_index_path):
        """

        :param segment_size:
        :param step_size:
        :param nb_classes:
        :param data_path:
        :param train_index_path:
        :param test_index_path:
        :param validation_index_path:
        :param standardization_paramters_path:
        :return:
        """

        # loading training/validation/test folds
        print('--------------------------------- Loading folds ------------------------------------------------')
        training_clips, train_clips_label = self.load_fold_with_labels('train', data_path, train_index_path)
        self.test_clips, self.test_clips_labels = self.load_fold_with_labels('test', data_path, test_index_path)
        validation_clips, validation_clips_label = self.load_fold_with_labels('validation', data_path,
                                                                              validation_index_path)

        # standardization
        print('-------------------------------- Standardization -----------------------------------------------')
        mean_vector, std_vector = self.retrieve_standardization_parameters(training_clips, train_index_path)
        training_clips = self.standardize_data(training_clips, mean_vector, std_vector)
        self.test_clips = self.standardize_data(self.test_clips, mean_vector, std_vector)
        validation_clips = self.standardize_data(validation_clips, mean_vector, std_vector)

        # segmentation
        print('--------------------------- Segments ( q ) Extraction -----------------------------------------')
        self.train_segments, self.train_labels = self.segment_fold('train', training_clips, train_clips_label,
                                                                   segment_size,
                                                                   step_size)
        self.test_segments, self.test_labels = self.segment_fold('test', self.test_clips, self.test_clips_labels,
                                                                 segment_size,
                                                                 step_size)
        self.validation_segments, self.validation_labels = self.segment_fold('validation', validation_clips,
                                                                             validation_clips_label,
                                                                             segment_size, step_size)

        # shuffle training data and its labels
        train_rand_index = np.random.permutation(self.train_labels.shape[0])
        self.train_segments = self.train_segments[train_rand_index, :, :]
        self.train_labels = self.train_labels[train_rand_index]

        # convert class vectors to binary class matrices
        self.train_one_hot_target = np_utils.to_categorical(self.train_labels, nb_classes)
        self.test_one_hot_target = np_utils.to_categorical(self.test_labels, nb_classes)
        self.validation_one_hot_target = np_utils.to_categorical(self.validation_labels, nb_classes)
