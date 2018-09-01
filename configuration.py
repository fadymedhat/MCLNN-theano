import os


class Configuration:
    USE_PRETRAINED_WEIGHTS = True  # True or False - no training is initiated (pre-trained weights are used)

    # IMPORTANT! This flag will affect training epochs.
    # Keep Visualization DISABLED if you need a properly trained model.
    # Visualization occurs in a callback function, which requires a call to model.predict(). It was found that calling the
    # prediction at the end of every epoch during training affects the weights for later epochs even if a new model is
    # loaded from the weights stored on from a previous epoch. You can check this behavior by tracking the validation
    # accuracy with the flag enabled or disabled. This could be related to the _Learning_phase flag in keras
    # (used to disable/enable dropout), which is not disabled when predicting during the training phase.
    # The below flag shows visualization for first MCLNN layer only.
    # This flag will affect training time and accuracy, DO NOT enable unless you only need to visualize the MCLNN layer predictions
    SAVE_SEGMENT_PREDICTION_IMAGE_PER_EPOCH = False  # True or False - store an image of the prediction of a specific segment at the end of every epoch.

    TRAIN_SEGMENT_INDEX = 500  # train segment index to plot during training
    TEST_SEGMENT_INDEX = 500  # test segment index to plot during training
    VALIDATION_SEGMENT_INDEX = 500  # validation segment index to plot during training

    NB_EPOCH = 2000  # maximum number of epochs
    WAIT_COUNT = 50  # early stopping count
    LEARNING_RATE = 0.0001
    SPLIT_COUNT = 3  # training/testing/validation splits
    TRAIN_FOLD_NAME = 'train'
    TEST_FOLD_NAME = 'test'
    VALIDATION_FOLD_NAME = 'validation'
    STOPPING_CRITERION = 'val_acc'  # 'val_acc' or 'val_loss'

    # the following flags are used during testing phase
    # they allow storing the predictions of n consecutive segments across all layers before the pooling layer.
    # Enabling the below flag will allow storing a visualization of the MCLNN weights.
    # Note! this is applied for the first fold only and disabled later on.
    SAVE_TEST_SEGMENT_PREDICTION_IMAGE = True  # True or False - store prediction images for segments of a specific clip of testing data
    SAVE_TEST_SEGMENT_PREDICTION_INITIAL_SEGMENT_INDEX = 50  # first segment to plot. This count is used only if the SAVE_LAYER_OUTPUT_IMAGE is enabled
    SAVE_TEST_SEGMENT_PREDICTION_IMAGE_COUNT = 30  # number of segments to save after the first segment. This count is used only if the SAVE_LAYER_OUTPUT_IMAGE is enabled
    HIDDEN_NODES_SLICES_COUNT = 40  # weights visualization for n hidden nodes


class ESC10(Configuration):
    # A model of 85.5% accuarcy
    DATASET_NAME = 'ESC10'
    CROSS_VALIDATION_FOLDS_COUNT = 5
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    PARENT_PATH = 'I:/ESC10-for-MCLNN'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'
    VISUALIZATION_PARENT_PATH = COMMON_PATH_NAME + '_visualization'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'esc10Specmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5')

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 600  # the samples in a mini-batch
    NB_CLASSES = 10  # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.5, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 200, 100, 100, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15, 15]  # the order for each layer
    MASK_BANDWIDTH = [20, 5]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-5, 3]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 40  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['DB', 'Ra', 'SW', 'BC', 'CT', 'PS', 'He', 'Ch', 'Ro', 'FC']


class ESC10AUGMENTED(Configuration):
    # A model of 85.25% accuarcy
    DATASET_NAME = 'ESC10_12augmentations'
    CROSS_VALIDATION_FOLDS_COUNT = 5
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    PARENT_PATH = 'I:/ESC10-augmented-for-MCLNN'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'
    VISUALIZATION_PARENT_PATH = COMMON_PATH_NAME + '_visualization'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'esc10aug_8pitch_4stretch_Specmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5')

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 600  # the samples in a mini-batch
    NB_CLASSES = 10  # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.5, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 200, 100, 100, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15, 15]  # the order for each layer
    MASK_BANDWIDTH = [20, 5]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-5, 3]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 20  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['DB', 'Ra', 'SW', 'BC', 'CT', 'PS', 'He', 'Ch', 'Ro', 'FC']


class ESC50(Configuration):
    # A model of 62.85% accuarcy

    DATASET_NAME = 'ESC50'
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    CROSS_VALIDATION_FOLDS_COUNT = 5
    PARENT_PATH = 'I:/ESC50-for-MCLNN'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'
    VISUALIZATION_PARENT_PATH = COMMON_PATH_NAME + '_visualization'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'esc50Specmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5')

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 300  # the samples in a mini-batch
    NB_CLASSES = 50  # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [14]  # the order for each layer
    MASK_BANDWIDTH = [20]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-5]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 40  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['Do', 'Ro', 'Pi', 'Cw', 'Fr', 'Ca', 'He', 'In', 'Sh', 'Cr',
                   'Ra', 'Sw', 'Cf', 'Ck', 'Cb', 'Wd', 'Wi', 'Pw', 'Tf', 'Th',
                   'Cy', 'Sn', 'Cl', 'Be', 'Cg', 'Fo', 'La', 'Bt', 'Sg',
                   'Ds', 'Dk', 'Mc', 'Kt', 'Dc', 'Co', 'Wm', 'Vc', 'Cm',
                   'Ct', 'Gb', 'Hp', 'Cs', 'Si', 'Ch', 'En', 'Tr', 'Cu', 'Ai', 'Fi', 'Hs']


class ESC50AUGMENTED(Configuration):
    # A model of 66.6% accuarcy
    DATASET_NAME = 'ESC50_4augmentations'
    CROSS_VALIDATION_FOLDS_COUNT = 5
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    PARENT_PATH = 'I:/ESC50-augmented-for-MCLNN'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'
    VISUALIZATION_PARENT_PATH = COMMON_PATH_NAME + '_visualization'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'esc50_4augmentationsSpecmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5')

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 300  # the samples in a mini-batch
    NB_CLASSES = 50  # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [14]  # the order for each layer
    MASK_BANDWIDTH = [20]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-5]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 40  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['Do', 'Ro', 'Pi', 'Cw', 'Fr', 'Ca', 'He', 'In', 'Sh', 'Cr',
                   'Ra', 'Sw', 'Cf', 'Ck', 'Cb', 'Wd', 'Wi', 'Pw', 'Tf', 'Th',
                   'Cy', 'Sn', 'Cl', 'Be', 'Cg', 'Fo', 'La', 'Bt', 'Sg',
                   'Ds', 'Dk', 'Mc', 'Kt', 'Dc', 'Co', 'Wm', 'Vc', 'Cm',
                   'Ct', 'Gb', 'Hp', 'Cs', 'Si', 'Ch', 'En', 'Tr', 'Cu', 'Ai', 'Fi', 'Hs']


class URBANSOUND8K(Configuration):
    # A model of 74.22% accuarcy
    DATASET_NAME = 'UrbanSound8K'
    CROSS_VALIDATION_FOLDS_COUNT = 10
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    PARENT_PATH = 'I:/UrbanSound8K-for-MCLNN'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'
    VISUALIZATION_PARENT_PATH = COMMON_PATH_NAME + '_visualization'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'urbansound8kSpecmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=1_FN=170_4secsDelta.hdf5')

    STEP_SIZE = 2  # overlap between segments is q minus step_size
    BATCH_SIZE = 500  # the samples in a mini-batch
    NB_CLASSES = 10  # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15]  # the order for each layer
    MASK_BANDWIDTH = [20]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-5]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 50  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['AC', 'CH', 'CP', 'DB', 'Dr', 'EI', 'GS', 'Ja', 'Si', 'SM']


class YORNOISE(Configuration):
    # A model of 75.82% accuarcy
    DATASET_NAME = 'YorNoise'
    CROSS_VALIDATION_FOLDS_COUNT = 10
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    PARENT_PATH = 'I:/YorNoise-for-MCLNN'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'
    VISUALIZATION_PARENT_PATH = COMMON_PATH_NAME + '_visualization'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'yornoiseSpecmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=1_FN=170_4secsDelta.hdf5')

    STEP_SIZE = 2  # overlap between segments is q minus step_size
    BATCH_SIZE = 500  # the samples in a mini-batch
    NB_CLASSES = 12  # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15]  # the order for each layer
    MASK_BANDWIDTH = [20]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-5]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 50  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['AC', 'CH', 'CP', 'DB', 'Dr', 'EI', 'GS', 'Ja', 'Si', 'SM', 'Ra', 'Tr']


class BALLROOM(Configuration):
    # A model of 92.55% accuarcy
    INITIAL_FOLD_ID = 0  # the initial fold to start with zero indexed. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/Ballroom-for-MCLNN/ballroom_10_folds_index'
    # DATASET_FILE_PATH = 'F:/Ballroom-for-MCLNN/ballroomSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=23_FN=600_30secs.hdf5'
    DATASET_FILE_PATH = 'F:/Ballroom-for-MCLNN/ballroomSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=23_FN_600_30sec.hdf5'
    # ALL_FOLDS_WEIGHTS_PATH = 'F:/Ballroom-for-MCLNN/ballroom_10_folds_pretrained_weights_92.55percent'
    # VISUALIZATION_PARENT_PATH = 'F:/Ballroom-for-MCLNN/ballroom_10_folds_visualization'

    ALL_FOLDS_WEIGHTS_PATH = 'I:\locallyconnectedlearnedweights/ballroomvis/ballroom_10_folds_pretrained_weights_92.55percentSegmentPlot'
    VISUALIZATION_PARENT_PATH = 'I:\locallyconnectedlearnedweights/ballroomvis/ballroom_10_folds_visualization'

    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 600  # the samples in a mini-batch
    NB_CLASSES = 8  # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.1, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 50, 10, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [20]  # the order for each layer
    MASK_BANDWIDTH = [40]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-10]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 55  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['CC', 'Ji', 'QS', 'Ru', 'Sa', 'Ta', 'VW', 'Wa']


class HOMBURG(Configuration):
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/Homburg-for-MCLNN/Homburg_10_folds_index/'
    DATASET_FILE_PATH = 'F:/Homburg-for-MCLNN/homburgSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=8_FN=200_10secs.hdf5'
    ALL_FOLDS_WEIGHTS_PATH = 'F:/Homburg-for-MCLNN/homburg_10_folds_pretrained_weights_61.45percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 800  # the samples in a mini-batch
    NB_CLASSES = 9  # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, 10, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [5, 5]  # the order for each layer
    MASK_BANDWIDTH = [40, 10]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-10, 3]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 1  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['Al', 'Bl', 'El', 'FC', 'FS', 'Ja', 'Po', 'RH', 'Ro']


class GTZAN(Configuration):
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/GTZAN-for-MCLNN/GTZAN_10_folds_index'
    DATASET_FILE_PATH = 'F:/GTZAN-for-MCLNN/gtzanSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=23_FN=600_30secs.hdf5'
    ALL_FOLDS_WEIGHTS_PATH = 'F:/GTZAN-for-MCLNN/GTZAN_10_folds_pretrained_weights_85percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 600  # the samples in a mini-batch
    NB_CLASSES = 10  # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 1  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [4, 4]  # the order for each layer
    MASK_BANDWIDTH = [40, 10]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-10, 3]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 10  # the k extra frames

    CLASS_NAMES = ['Bl', 'Cl', 'Co', 'Di', 'Hi', 'Ja', 'Me', 'Po', 'Re', 'Ro']


class ISMIR2004(Configuration):
    INITIAL_FOLD_ID = 0  # the initial fdold to start with. This should be zero unless you want to start from another fold
    DATASET_FILE_PATH = 'F:/ISMIR2004-for-MCLNN/ismir2004Specmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=600_FN=600_30secs.hdf5'
    INDEX_PATH = 'F:/ISMIR2004-for-MCLNN/ISMIR2004_10_folds_index_1754157958_hdf5'
    ALL_FOLDS_WEIGHTS_PATH = 'F:/ISMIR2004-for-MCLNN/ISMIR2004_10_folds_pretrained_weights_85.0percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 600  # the samples in a mini-batch
    NB_CLASSES = 6  # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 1  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [4, 4]  # the order for each layer
    MASK_BANDWIDTH = [40, 10]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-10, 3]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 10  # the k extra frames

    CLASS_NAMES = ['Cl', 'El', 'Ja', 'Me', 'Po', 'Wo']
