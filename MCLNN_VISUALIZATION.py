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
from parameters import ESC10, ESC50, URBANSOUND8K, BALLROOM, GTZAN, HOMBURG, ISMIR2004, YORNOISE, ESC10AUGMENTED, ESC50AUGMENTED
from keras import callbacks
from sklearn.metrics import f1_score as f1score
from sklearn.metrics import confusion_matrix


Config = BALLROOM # 92.11% Checked-Weights upload - Fold name check
# Config = HOMBURG # 61.45% Checked - weights to upload - Fold name check
# Config = ESC50 # 62.85% Checked - weights to upload - Fold name check
# Config = ESC10 # 85.5% Checked - weights upload - Fold name check
# Config = GTZAN # 85% majority vote Checked-weights to upload
# Config = YORNOISE # 75.816 - weights upload - Fold name check
# Config = URBANSOUND8K # 74.37% Checked- weights upload
# Config = ISMIR2004 # 86% majority vote,

# git config --global credential.helper wincred

# Config = ESC10AUGMENTED #
# Config = ESC50AUGMENTED

segment_size = sum(Config.LAYERS_ORDER_LIST) * 2 + Config.EXTRA_FRAMES
folds_index_file_list = glob.glob(os.path.join(Config.INDEX_PATH, "Fold*.hdf5"))
folds_index_file_list.sort()
cross_val_index_list = np.arange(0, Config.SPLIT_COUNT * Config.CROSS_VALIDATION_FOLDS_COUNT, Config.SPLIT_COUNT)

test_index_path = folds_index_file_list[j]
train_index_path = folds_index_file_list[j + 1]
validation_index_path = folds_index_file_list[j + 2]

np.random.seed(0)  # for reproducibility

train_segments, train_labels, \
test_segments, test_labels, \
validation_Segments, validation_labels, \
test_clips, fold_target_label = DataLoader.load_data(
    segment_size, Config.STEP_SIZE, Config.FILE_PATH, train_index_path, test_index_path, validation_index_path)


# convert class vectors to binary class matrices
train_one_hot_target = np_utils.to_categorical(train_labels, Config.NB_CLASSES)
test_one_hot_target = np_utils.to_categorical(test_labels, Config.NB_CLASSES)
validation_one_hot_target = np_utils.to_categorical(validation_labels, Config.NB_CLASSES)

def evaluate_model(model):
    # ________________ Frame level evaluation for Test/Validation splits ________________________
    print ('Validation segments = ' + str(validation_Segments.shape) +
           ' one-hot encoded target' + str(validation_one_hot_target.shape))
    score = model.evaluate(validation_Segments, validation_one_hot_target, verbose=0)
    print('Validation score:', score[0])
    print('Validation accuracy:', score[1])

    print('Test segments = ' + str(test_segments.shape) +
          ' one-hot encoded target' + str(test_one_hot_target.shape))
    score = model.evaluate(test_segments, test_one_hot_target, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # ___________________ predict frame-level classes ___________________________________
    segments_predicted_labels = np_utils.categorical_probas_to_classes(model.predict(test_segments))
    test_true_frame_label = np_utils.categorical_probas_to_classes(test_one_hot_target)

    cm_frames = confusion_matrix(test_true_frame_label, segments_predicted_labels)
    # np.set_printoptions(precision=2)
    print('Confusion matrix, frame level')
    # cm_frames = np.transpose(cm_frames)
    print(cm_frames)
    print('Frame level accuracy :' + str(np_utils.accuracy(segments_predicted_labels, test_true_frame_label)))

    # ----------------
    from keras.models import Model
    modelIntermediate = Model(input=model.input,
                              output=model.get_layer(
                                  'prelu0').output)  # block3_flatter block4_prelu block5_dropout

    # -------------



    # -------------- Voting ------------------------
    clipPredictedProbabilitySumVote = []
    clip_predicted_probability_mean_vote = []
    clip_predicted_majority_vote = []
    for i, clip in enumerate(test_clips):
        segments, segments_target_labels = DataLoader.segment_clip(clip, fold_target_label[i], segment_size,
                                                                   step_size=Config.STEP_SIZE)
        segments_predicted_prop = model.predict(segments)
        segments_predicted_labels = np_utils.categorical_probas_to_classes(segments_predicted_prop)
        labels_histogram = np.bincount(segments_predicted_labels)
        clip_predicted_majority_vote.append(np.argmax(labels_histogram))
        # clipPredictedProbabilitySumVote.append(np.argmax(np.sum(segments_predicted_prop, axis=0)))
        clip_predicted_probability_mean_vote.append(np.argmax(np.mean(segments_predicted_prop, axis=0)))

    # ----------------------------------------------

    # block4_prelu = modelIntermediate.predict(segments)
    #
    # import matplotlib.pyplot as plt
    # plt.figure(8)
    # h = modelIntermediate.get_weights()[0]
    # imgplot = plt.imshow(h[0, :, :])
    # plt.show()
    # imgplot = plt.imshow(np.transpose(block4_prelu[40000:40200, :]))

    from keras.models import Model

    modelIntermediate_m = Model(input=model.input,
                                output=model.get_layer(
                                    'mclnn0').output)
    block4_prelu_test_m = modelIntermediate_m.predict(test_segments)

    modelIntermediate_s = Model(input=model.input,
                                output=model.get_layer(
                                    'prelu0').output)
    block4_prelu_test_s = modelIntermediate_s.predict(test_segments)

    modelIntermediate_f = Model(input=model.input,
                                output=model.get_layer(
                                    'flatter0').output)  # block1_mclnn block3_flatter block4_prelu block5_dropout
    block4_prelu_test_f = modelIntermediate_f.predict(test_segments)
    # block4_prelu = modelIntermediate.predict(X_train)


import matplotlib.pyplot as plt

plt.figure(8)
# h = modelIntermediate_f.get_weights()[0]
# imgplot = plt.imshow(h[0, :, :])
# imgplot = plt.imshow(np.transpose(block4_prelu_test_m[300, :, :]))
# imgplot = plt.imshow(np.transpose(block4_prelu_f[40000:40200, :]))

plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    left='off',
    right='off',
    labelbottom='off',
    labelleft='off')  # labels along the bottom edge are off

plt.tick_params(
    axis='both',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    left='off',
    right='off',
    labelbottom='off',
    labelleft='off')

for i in range(0, 600, 50):
    imgplot = plt.imshow(np.transpose(block4_prelu_test_s[505 * 65 + i, :, :]), cmap='gray', vmin=0, vmax=1)
    # imgplot = plt.imshow(np.transpose(block4_prelu_test_s[i, :, :]), cmap='gray', vmin=0, vmax=1)
    plt.savefig('I:/locallyconnectedlearnedweights/visual_mclnn/mclnntest' + str(i) + '.png', dpi=1200,
                bbox_inches=None)
    plt.savefig('I:/locallyconnectedlearnedweights/visual_mclnn/mclnntest' + str(i) + '.svg', dpi=1200,
                bbox_inches=None)
    plt.savefig('I:/locallyconnectedlearnedweights/visual_mclnn/mclnntest' + str(i) + '.eps', dpi=1200,
                bbox_inches=None)

# block4_prelu = modelIntermediate.predict(segments)
#
# import matplotlib.pyplot as plt
# plt.figure(8)
# h = modelIntermediate.get_weights()[0]
# imgplot = plt.imshow(h[0, :, :])
# plt.show()
# imgplot = plt.imshow(np.transpose(block4_prelu[40000:40200, :]))

from keras.models import Model

modelIntermediate_m = Model(input=model.input,
                            output=model.get_layer(
                                'mclnn0').output)
block4_prelu_test_m = modelIntermediate_m.predict(test_segments)

modelIntermediate_s = Model(input=model.input,
                            output=model.get_layer(
                                'prelu0').output)
block4_prelu_test_s = modelIntermediate_s.predict(test_segments)

modelIntermediate_f = Model(input=model.input,
                            output=model.get_layer(
                                'flatter0').output)  # block1_mclnn block3_flatter block4_prelu block5_dropout
block4_prelu_test_f = modelIntermediate_f.predict(test_segments)
# block4_prelu = modelIntermediate.predict(X_train)


# ----------------------------------------------





