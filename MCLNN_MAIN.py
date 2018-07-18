'''
Masked ConditionaL Neural Networks
'''
from __future__ import print_function
import gc
import os
import glob
import matplotlib.cm as cm
import datetime
import numpy as np
import numpy.ma as ma
import pylab as pl
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
from trainingcallbacks import DirectoryHouseKeepingCallback
from datapreprocessor import DataLoader
from layers import MaskedConditional, GlobalPooling1D
import platform as plf
from parameters import ESC10, ESC50, URBANSOUND8K, BALLROOM, GTZAN, HOMBURG, ISMIR2004, YORNOISE, ESC10AUGMENTED, \
    ESC50AUGMENTED
from keras import callbacks
from sklearn.metrics import f1_score as f1score
from sklearn.metrics import confusion_matrix

from memory_profiler import profile

Config = ESC10  # 85.5% Checked - weights upload - Fold name check
# Config = URBANSOUND8K # 74.22% NEW trans script
# Config = BALLROOM # 92.55% NEW trans script
# Config = YORNOISE # 75.816 - NEW trans script
# Config = ISMIR2004 # 85% majority vote, NEW trans script
# Config = HOMBURG # 61.45% Checked - weights to upload - Fold name check
# Config = GTZAN # 85% majority vote Checked-weights to upload
# Config = ESC50 # 62.85% Checked - weights to upload - Fold name check
# Config = ESC50AUGMENTED # 66.85%
# Config = ESC10AUGMENTED # 85.25% Checked -weights upload
# JULY revision and final uploads check----------------------------










# -------------- END of july check --------------------------

USE_PRETRAINED_WEIGHTS = True  # True or False - no training is initiated (pre-trained weights are used)


class MCLNNTrainer(object):
    def build_model(self, segment_size, feature_count, pretrained_weights_path=None):

        model = Sequential()

        layer_index = 0

        for layer_index in range(Config.MCLNN_LAYER_COUNT):
            print('Layer' + str(layer_index) + ' - Dropout = ' + str(Config.DROPOUT[layer_index]) +
                  ', Initialization = ' + str(Config.WEIGHT_INITIALIZATION[layer_index]) +
                  ', Order = ' + str(Config.LAYERS_ORDER_LIST[layer_index]) +
                  ', Bandwidth = ' + str(Config.MASK_BANDWIDTH[layer_index]) +
                  ', Overlap = ' + str(Config.MASK_OVERLAP[layer_index]) +
                  ', Hidden nodes = ' + str(Config.HIDDEN_NODES_LIST[layer_index]))
            model.add(Dropout(Config.DROPOUT[layer_index],
                              input_shape=(segment_size, feature_count),
                              name='dropout' + str(layer_index)))

            model.add(MaskedConditional(init=Config.WEIGHT_INITIALIZATION[layer_index],
                                        output_dim=Config.HIDDEN_NODES_LIST[layer_index],
                                        order=Config.LAYERS_ORDER_LIST[layer_index],
                                        bandwidth=Config.MASK_BANDWIDTH[layer_index],
                                        overlap=Config.MASK_OVERLAP[layer_index],
                                        name='mclnn' + str(layer_index)))
            model.add(PReLU(init='zero', weights=None, name='prelu' + str(layer_index)))

        model.add(GlobalPooling1D(name='globalpool' + str(layer_index)))
        model.add(Flatten(name='flatter' + str(layer_index)))

        # --------- Dense LAYER -----------------
        layer_index += 1
        for layer_index in range(layer_index, layer_index + Config.DENSE_LAYER_COUNT):
            print('Layer' + str(layer_index) + ' - Dropout = ' + str(Config.DROPOUT[layer_index]) +
                  ', Initialization = ' + str(Config.WEIGHT_INITIALIZATION[layer_index]) +
                  ', Hidden nodes = ' + str(Config.HIDDEN_NODES_LIST[layer_index]))
            model.add(Dropout(Config.DROPOUT[layer_index], name='dropout' + str(layer_index)))
            model.add(Dense(init=Config.WEIGHT_INITIALIZATION[layer_index],
                            output_dim=Config.HIDDEN_NODES_LIST[layer_index],
                            name='dense' + str(layer_index)))
            model.add(PReLU(name='prelu' + str(layer_index)))

        # --------- Output LAYER -----------------
        layer_index += 1
        print('Layer' + str(layer_index) + ' - Dropout = ' + str(Config.DROPOUT[layer_index]) +
              ', Initialization = ' + str(Config.WEIGHT_INITIALIZATION[layer_index]) +
              ', Hidden nodes = ' + str(Config.HIDDEN_NODES_LIST[layer_index]))
        model.add(Dropout(Config.DROPOUT[layer_index], name='dropout' + str(layer_index)))
        model.add(Dense(init=Config.WEIGHT_INITIALIZATION[layer_index],
                        output_dim=Config.HIDDEN_NODES_LIST[layer_index],
                        name='dense' + str(layer_index)))
        model.add(Activation('softmax', name='softmax' + str(layer_index)))

        model.summary()
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        if pretrained_weights_path != None:
            model.load_weights(pretrained_weights_path)

        return model

    def train_model(self, model, data_loader, weights_to_store_path):
        # stoppingCriterion = 'val_acc'  # 'val_acc' or 'val_loss'
        # print('----------- Monitor --------------- : ', stoppingCriterion)

        print('----------- waitCount --------------- : ', str(Config.WAIT_COUNT))

        # remote = callbacks.RemoteMonitor(root='http://localhost:9000')
        earlyStopping = callbacks.EarlyStopping(monitor=Config.STOPPING_CRITERION, patience=Config.WAIT_COUNT,
                                                verbose=0,
                                                mode='auto')
        weights_file_name_format = 'weights.epoch{epoch:02d}-val_loss{val_loss:.2f}-val_acc{val_acc:.4f}.hdf5'
        checkpoint = ModelCheckpoint(os.path.join(weights_to_store_path, weights_file_name_format),
                                     monitor='val_loss', verbose=0,
                                     save_best_only=False, mode='auto')
        directoryHouseKeeping = DirectoryHouseKeepingCallback(weights_to_store_path)

        before = datetime.datetime.now()
        print(before)
        history = model.fit(data_loader.train_segments, data_loader.train_one_hot_target,
                            batch_size=Config.BATCH_SIZE, nb_epoch=Config.NB_EPOCH,
                            verbose=1,
                            validation_data=(data_loader.validation_Segments, data_loader.validation_one_hot_target),
                            callbacks=[earlyStopping, checkpoint, directoryHouseKeeping])  # , remote,
        after = datetime.datetime.now()
        print(after)
        print('It took:')
        print(after - before)

    def evaluate_model(self, segment_size, model, data_loader):

        # # convert class vectors to binary class matrices
        # train_one_hot_target = np_utils.to_categorical(data_loader.train_labels, Config.NB_CLASSES)
        # test_one_hot_target = np_utils.to_categorical(data_loader.test_labels, Config.NB_CLASSES)
        # validation_one_hot_target = np_utils.to_categorical(data_loader.validation_labels, Config.NB_CLASSES)

        # ________________ Frame level evaluation for Test/Validation splits ________________________
        print('Validation segments = ' + str(data_loader.validation_segments.shape) +
              ' one-hot encoded target' + str(data_loader.validation_one_hot_target.shape))
        score = model.evaluate(data_loader.validation_segments, data_loader.validation_one_hot_target, verbose=0)
        print('Validation score:', score[0])
        print('Validation accuracy:', score[1])

        print('Test segments = ' + str(data_loader.test_segments.shape) +
              ' one-hot encoded target' + str(data_loader.test_one_hot_target.shape))
        score = model.evaluate(data_loader.test_segments, data_loader.test_one_hot_target, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # ___________________ predict frame-level classes ___________________________________
        test_predicted_labels = np_utils.categorical_probas_to_classes(model.predict(data_loader.test_segments))
        test_target_labels = np_utils.categorical_probas_to_classes(data_loader.test_one_hot_target)

        cm_frames = confusion_matrix(test_target_labels, test_predicted_labels)
        print('Confusion matrix, frame level')
        print(cm_frames)
        print('Frame level accuracy :' + str(np_utils.accuracy(test_predicted_labels, test_target_labels)))

        # -------------- Voting ------------------------
        clip_predicted_probability_mean_vote = []
        clip_predicted_majority_vote = []
        for i, clip in enumerate(data_loader.test_clips):
            segments, segments_target_labels = data_loader.segment_clip(data=clip,
                                                                        label=data_loader.test_clips_labels[i],
                                                                        segment_size=segment_size,
                                                                        step_size=Config.STEP_SIZE)
            segments_predicted_prop = model.predict(segments)
            test_predicted_labels = np_utils.categorical_probas_to_classes(segments_predicted_prop)
            labels_histogram = np.bincount(test_predicted_labels)
            clip_predicted_majority_vote.append(np.argmax(labels_histogram))
            clip_predicted_probability_mean_vote.append(np.argmax(np.mean(segments_predicted_prop, axis=0)))

        cm_majority = confusion_matrix(data_loader.test_clips_labels, clip_predicted_majority_vote)
        print('Fold Confusion matrix - Majority voting - Clip level :')
        print(Config.CLASS_NAMES)
        print(cm_majority)
        print('Clip-level majority-vote Accuracy ' + str(
            np_utils.accuracy(clip_predicted_majority_vote, data_loader.test_clips_labels)))

        print('Fold Confusion matrix - Probability MEAN voting - Clip level :')
        cm_probability = confusion_matrix(data_loader.test_clips_labels, clip_predicted_probability_mean_vote)
        print(Config.CLASS_NAMES)
        print(cm_probability)
        print('Clip-level probability-vote Accuracy ' + str(
            np_utils.accuracy(np.asarray(clip_predicted_probability_mean_vote),
                              np.squeeze(data_loader.test_clips_labels))))

        scoref1 = f1score(data_loader.test_clips_labels, clip_predicted_probability_mean_vote, average='micro')
        print('F1 Score micro ' + str(scoref1))

        scoref1 = f1score(data_loader.test_clips_labels, clip_predicted_probability_mean_vote, average='weighted')
        print('F1 Score weighted ' + str(scoref1))

        return cm_majority, cm_probability, clip_predicted_majority_vote, clip_predicted_probability_mean_vote, data_loader.test_clips_labels


@profile
def run():
    all_folds_target_label = np.asarray([])

    all_folds_majority_vote_cm = np.zeros((Config.NB_CLASSES, Config.NB_CLASSES), dtype=np.int)
    all_folds_majority_vote_label = np.asarray([])

    all_folds_probability_vote_cm = np.zeros((Config.NB_CLASSES, Config.NB_CLASSES), dtype=np.int)
    all_folds_probability_vote_label = np.asarray([])

    segment_size = sum(Config.LAYERS_ORDER_LIST) * 2 + Config.EXTRA_FRAMES
    print('window:' + str(segment_size))

    # list of paths to the n-fold indices of the Training/Testing/Validation splits
    # number of paths should be e.g. 30 for 3x10, where 3 is for the splits and 10 for the 10-folds
    # Every 3 files are for one run to train and validate on 9-folds and test on the remaining fold.
    folds_index_file_list = glob.glob(os.path.join(Config.INDEX_PATH, "Fold*.hdf5"))
    folds_index_file_list.sort()

    cross_val_index_list = np.arange(0, Config.SPLIT_COUNT * Config.CROSS_VALIDATION_FOLDS_COUNT, Config.SPLIT_COUNT)

    for j in range(cross_val_index_list[Config.INITIAL_FOLD_ID], len(folds_index_file_list), Config.SPLIT_COUNT):

        test_index_path = folds_index_file_list[j] if folds_index_file_list[j].lower().endswith(
            '_test.hdf5') else None
        train_index_path = folds_index_file_list[j + 1] if folds_index_file_list[j + 1].lower().endswith(
            '_train.hdf5') else None
        validation_index_path = folds_index_file_list[j + 2] if folds_index_file_list[j + 2].lower().endswith(
            '_validation.hdf5') else None


        if None in [test_index_path, train_index_path, validation_index_path]:
            print('Train / Validation / Test indices are not correctly assigned')
            exit(1)

        np.random.seed(0)  # for reproducibility

        data_loader = DataLoader()
        mclnn_trainer = MCLNNTrainer()

        data_loader.load_data(segment_size,
                              Config.STEP_SIZE,
                              Config.NB_CLASSES,
                              Config.FILE_PATH,
                              train_index_path,
                              test_index_path,
                              validation_index_path)

        train_index_filename = os.path.basename(train_index_path).split('.')[0]
        weights_to_store_foldername = train_index_filename + '_' \
                                      + 'batch' + str(Config.BATCH_SIZE) \
                                      + 'wait' + str(Config.WAIT_COUNT) \
                                      + 'order' + str(Config.LAYERS_ORDER_LIST[0]) \
                                      + 'extra' + str(Config.EXTRA_FRAMES)
        weights_to_store_path = os.path.join(Config.WEIGHTS_TO_STORE_PATH, weights_to_store_foldername)
        if not os.path.exists(weights_to_store_path):
            os.makedirs(weights_to_store_path)

        print('----------- Training param -------------')
        print(' batch_size>' + str(Config.BATCH_SIZE) +
              ' nb_classes>' + str(Config.NB_CLASSES) +
              ' nb_epoch>' + str(Config.NB_EPOCH) +
              ' mclnn_layers>' + str(Config.MCLNN_LAYER_COUNT) +
              ' dense_layers>' + str(Config.DENSE_LAYER_COUNT) +
              ' norder>' + str(Config.LAYERS_ORDER_LIST) +
              ' extra_frames>' + str(Config.EXTRA_FRAMES) +
              ' segment_size>' + str(segment_size + 1) +  # plus 1 is for middle frame, considered in segmentation stage
              ' initial_fold>' + str(Config.INITIAL_FOLD_ID + 1) +  # plus 1 beacuse folds are zero indexed
              ' wait_count>' + str(Config.WAIT_COUNT) +
              ' split_count>' + str(Config.SPLIT_COUNT))

        if USE_PRETRAINED_WEIGHTS == False:
            model = mclnn_trainer.build_model(segment_size=data_loader.train_segments.shape[1],
                                              feature_count=data_loader.train_segments.shape[2],
                                              pretrained_weights_path=startup_weights)
            mclnn_trainer.train_model(model, data_loader, weights_to_store_path)

        # load paths of all weights generated during training
        weight_list = glob.glob(os.path.join(weights_to_store_path, "*.hdf5"))
        weight_list.sort(key=os.path.getmtime)

        if len(weight_list) > 1:
            startup_weights = weight_list[-(Config.WAIT_COUNT + 2)]
        elif len(weight_list) == 1:
            startup_weights = weight_list[0]
        print('----------- Weights Loaded ---------------:')
        print(startup_weights)

        model = mclnn_trainer.build_model(segment_size=data_loader.train_segments.shape[1],
                                          feature_count=data_loader.train_segments.shape[2],
                                          pretrained_weights_path=startup_weights)

        fold_majority_cm, fold_probability_cm, \
        fold_majority_vote_label, fold_probability_vote_label, \
        fold_target_label = mclnn_trainer.evaluate_model(segment_size=segment_size,
                                                         model=model,
                                                         data_loader=data_loader)

        all_folds_majority_vote_cm += fold_majority_cm
        all_folds_majority_vote_label = np.append(all_folds_majority_vote_label, fold_majority_vote_label)
        all_folds_probability_vote_cm += fold_probability_cm
        all_folds_probability_vote_label = np.append(all_folds_probability_vote_label, fold_probability_vote_label)

        all_folds_target_label = np.append(all_folds_target_label, fold_target_label)

        gc.collect()

    print('-------------- Cross validation performance --------------')

    print(Config.CLASS_NAMES)
    print(all_folds_majority_vote_cm)
    print(str(Config.CROSS_VALIDATION_FOLDS_COUNT) + '-Fold Clip-level majority-vote Accuracy ' + str(
        np_utils.accuracy(all_folds_majority_vote_label, all_folds_target_label)))

    print(Config.CLASS_NAMES)
    print(all_folds_probability_vote_cm)
    print(str(Config.CROSS_VALIDATION_FOLDS_COUNT) + '-Fold Clip-level probability-vote Accuracy ' + str(
        np_utils.accuracy(all_folds_probability_vote_label, all_folds_target_label)))

    scoref1 = f1score(all_folds_target_label, all_folds_probability_vote_label, average='micro')
    print('F1 Score micro ' + str(scoref1))

    scoref1 = f1score(all_folds_target_label, all_folds_probability_vote_label, average='weighted')
    print('F1 Score weighted ' + str(scoref1))


if __name__ == "__main__":
    run()





    # W = model.layers[1].W.get_value(borrow=True)
    # W = np.squeeze(W)
    # print("W shape : ", W.shape)
    # pl.figure(figsize=(15, 15))
    # pl.title('conv1 weights')
    # nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)






    # def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    #     """Wrapper around pl.imshow"""
    #     if cmap is None:
    #         cmap = cm.jet
    #     if vmin is None:
    #         vmin = data.min()
    #     if vmax is None:
    #         vmax = data.max()
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    #     pl.colorbar(im, cax=cax)
    #
    # def make_mosaic(imgs, nrows, ncols, border=1):
    #     """
    #     Given a set of images with all the same shape, makes a
    #     mosaic with nrows and ncols
    #     """
    #     nimgs = imgs.shape[0]
    #     imshape = imgs.shape[1:]
    #
    #     mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
    #                             ncols * imshape[1] + (ncols - 1) * border),
    #                            dtype=np.float32)
    #
    #     paddedh = imshape[0] + border
    #     paddedw = imshape[1] + border
    #     for i in xrange(nimgs):
    #         row = int(np.floor(i / ncols))
    #         col = i % ncols
    #
    #         mosaic[row * paddedh:row * paddedh + imshape[0],
    #         col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    #     return mosaic
