
class Configuration:
    NB_EPOCH = 2000 # maximum number of epochs
    WAIT_COUNT = 50 # early stopping count
    SPLIT_COUNT = 3 # training/testing/validation splits
    TRAIN_FOLD_NAME = 'train'
    TEST_FOLD_NAME = 'test'
    VALIDATION_FOLD_NAME = 'validation'
    STOPPING_CRITERION = 'val_acc'  # 'val_acc' or 'val_loss'

class ESC10(Configuration):
    # A model of 85.5% accuarcy
    INITIAL_FOLD_ID = 0 # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/ESC10-for-MCLNN/ESC10_5_folds_index'
    FILE_PATH = 'F:/ESC10-for-MCLNN/esc10Specmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/ESC10-for-MCLNN/ESC10_5_folds_pretrained_weights_85.5percent'
    CROSS_VALIDATION_FOLDS_COUNT = 5

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 600 # the samples in a mini-batch
    NB_CLASSES = 10 # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.5, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 200, 100, 100, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15, 15]  # the order for each layer
    MASK_BANDWIDTH = [20, 5] # the consecutive features enabled at the input
    MASK_OVERLAP = [-5, 3] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 40  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['DB', 'Ra', 'SW', 'BC', 'CT', 'PS', 'He', 'Ch', 'Ro', 'FC']

class ESC10AUGMENTED(Configuration):

    INITIAL_FOLD_ID = 0 # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/ESC10-augmented-for-MCLNN/ESC10_5_folds_12augment_index'
    FILE_PATH = 'F:/ESC10-augmented-for-MCLNN/esc10aug_8pitch_4stretch_Specmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/ESC10-augmented-for-MCLNN/ESC10_5_folds_12augment_pretrained_weights_85.25percent'
    CROSS_VALIDATION_FOLDS_COUNT = 5

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 600 # the samples in a mini-batch
    NB_CLASSES = 10 # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.5, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 200, 100, 100, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15, 15]  # the order for each layer
    MASK_BANDWIDTH = [20, 5] # the consecutive features enabled at the input
    MASK_OVERLAP = [-5, 3] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 20  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['DB', 'Ra', 'SW', 'BC', 'CT', 'PS', 'He', 'Ch', 'Ro', 'FC']

class ESC50(Configuration):
    # A model of 62.85% accuarcy
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/ESC50-for-MCLNN/ESC50_5_folds_index'
    FILE_PATH = 'F:/ESC50-for-MCLNN/esc50Specmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/ESC50-for-MCLNN/ESC50_5_folds_pretrained_weights_62.85percent'
    CROSS_VALIDATION_FOLDS_COUNT = 5

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 300 # the samples in a mini-batch
    NB_CLASSES = 50 # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal',  'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [14]  # the order for each layer
    MASK_BANDWIDTH = [20] # the consecutive features enabled at the input
    MASK_OVERLAP = [-5] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 40  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['Do','Ro','Pi','Cw','Fr','Ca','He','In','Sh','Cr',
                    'Ra','Sw','Cf','Ck','Cb','Wd','Wi','Pw','Tf','Th',
                    'Cy','Sn','Cl','Be','Cg','Fo','La','Bt','Sg',
                    'Ds','Dk','Mc','Kt','Dc','Co','Wm','Vc','Cm',
                    'Ct','Gb','Hp','Cs','Si','Ch','En','Tr','Cu','Ai','Fi','Hs']

class ESC50AUGMENTED(Configuration):

    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/ESC50-augmented-for-MCLNN/ESC50_5_folds_4augment_index'
    FILE_PATH = 'F:/ESC50-augmented-for-MCLNN/esc50aug_2pitch_2stretch_Specmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=4_FN=200_5secsDelta.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/ESC50-augmented-for-MCLNN/ESC50_5_folds_4augment_pretrained_weights_66.85percent'
    CROSS_VALIDATION_FOLDS_COUNT = 5

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 300 # the samples in a mini-batch
    NB_CLASSES = 50 # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal',  'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [14]  # the order for each layer
    MASK_BANDWIDTH = [20] # the consecutive features enabled at the input
    MASK_OVERLAP = [-5] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 40  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['Do','Ro','Pi','Cw','Fr','Ca','He','In','Sh','Cr',
                    'Ra','Sw','Cf','Ck','Cb','Wd','Wi','Pw','Tf','Th',
                    'Cy','Sn','Cl','Be','Cg','Fo','La','Bt','Sg',
                    'Ds','Dk','Mc','Kt','Dc','Co','Wm','Vc','Cm',
                    'Ct','Gb','Hp','Cs','Si','Ch','En','Tr','Cu','Ai','Fi','Hs']

class URBANSOUND8K(Configuration):
    # A model of 74.37% accuarcy
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/UrbanSound8K-for-MCLNN/UrbanSound8K_10_folds_index'
    FILE_PATH = 'F:/UrbanSound8K-for-MCLNN/urbansound8kSpecmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=1_FN=170_4secsDelta.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/UrbanSound8K-for-MCLNN/UrbanSound8K_folds_pretrained_weights_74.37percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 2 # overlap between segments is q minus step_size
    BATCH_SIZE = 500 # the samples in a mini-batch
    NB_CLASSES = 10 # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15]  # the order for each layer
    MASK_BANDWIDTH = [20] # the consecutive features enabled at the input
    MASK_OVERLAP = [-5] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 50  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['AC', 'CH' , 'CP', 'DB', 'Dr', 'EI' , 'GS' , 'Ja' , 'Si' ,'SM']

class BALLROOM(Configuration):
    # A model of 92.55% accuarcy
    INITIAL_FOLD_ID = 0  # the initial fold to start with zero indexed. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/Ballroom-for-MCLNN/ballroom_10_folds_index'
    FILE_PATH = 'F:/Ballroom-for-MCLNN/ballroomSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=23_FN=600_30secs.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/Ballroom-for-MCLNN/ballroom_10_folds_pretrained_weights_92.55percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 600 # the samples in a mini-batch
    NB_CLASSES = 8 # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.1, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 50, 10, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal',  'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [20]  # the order for each layer
    MASK_BANDWIDTH = [40] # the consecutive features enabled at the input
    MASK_OVERLAP = [-10] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 55  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['CC', 'Ji', 'QS', 'Ru', 'Sa', 'Ta', 'VW', 'Wa']

class HOMBURG(Configuration):

    INITIAL_FOLD_ID = 0 # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/Homburg-for-MCLNN/Homburg_10_folds_index/'
    FILE_PATH = 'F:/Homburg-for-MCLNN/homburgSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=8_FN=200_10secs.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/Homburg-for-MCLNN/homburg_10_folds_pretrained_weights_61.45percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 800 # the samples in a mini-batch
    NB_CLASSES = 9 # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, 10, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [5, 5]  # the order for each layer
    MASK_BANDWIDTH = [40, 10] # the consecutive features enabled at the input
    MASK_OVERLAP = [-10, 3] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 1  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['Al', 'Bl' , 'El', 'FC', 'FS' , 'Ja' , 'Po' , 'RH' ,'Ro']

class GTZAN(Configuration):

    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/GTZAN-for-MCLNN/GTZAN_10_folds_index'
    FILE_PATH = 'F:/GTZAN-for-MCLNN/gtzanSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=23_FN=600_30secs.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/GTZAN-for-MCLNN/GTZAN_10_folds_pretrained_weights_85percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 600 # the samples in a mini-batch
    NB_CLASSES = 10 # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 1   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [4, 4]  # the order for each layer
    MASK_BANDWIDTH = [40, 10] # the consecutive features enabled at the input
    MASK_OVERLAP = [-10, 3] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 10  # the k extra frames

    CLASS_NAMES = ['Bl', 'Cl' , 'Co', 'Di', 'Hi', 'Ja' , 'Me' , 'Po' , 'Re' ,'Ro']

class ISMIR2004(Configuration):

    INITIAL_FOLD_ID = 0  # the initial fdold to start with. This should be zero unless you want to start from another fold
    FILE_PATH = 'F:/ISMIR2004-for-MCLNN/ismir2004Specmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=600_FN=600_30secs.hdf5'
    INDEX_PATH = 'F:/ISMIR2004-for-MCLNN/ISMIR2004_10_folds_index_1754157958_hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/ISMIR2004-for-MCLNN/ISMIR2004_10_folds_pretrained_weights_85.0percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 1 # overlap between segments is q minus step_size
    BATCH_SIZE = 600 # the samples in a mini-batch
    NB_CLASSES = 6 # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 1   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [4, 4]  # the order for each layer
    MASK_BANDWIDTH = [40, 10] # the consecutive features enabled at the input
    MASK_OVERLAP = [-10, 3] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 10  # the k extra frames

    CLASS_NAMES = ['Cl', 'El' , 'Ja', 'Me', 'Po', 'Wo' ]

class YORNOISE(Configuration):

    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    INDEX_PATH = 'F:/YorNoise-for-MCLNN/YorNoise_10_folds_index'
    FILE_PATH = 'F:/YorNoise-for-MCLNN/yornoiseSpecmeln_mels=60_nfft=1024_hoplength=512_fmax=NIL_22050hzsampling_FF=1_FN=170_4secsDelta.hdf5'
    WEIGHTS_TO_STORE_PATH = 'F:/YorNoise-for-MCLNN/YorNoise_10_folds_pretrained_weights_75.8percent'
    CROSS_VALIDATION_FOLDS_COUNT = 10

    STEP_SIZE = 2 # overlap between segments is q minus step_size
    BATCH_SIZE = 500 # the samples in a mini-batch
    NB_CLASSES = 12 # the number of classes to classify
    DROPOUT = [0.01, 0.5, 0.5, 0.1] # dropout at the input of each layer
    HIDDEN_NODES_LIST = [300, 100, 100, NB_CLASSES] # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 1  # number of MCLNN layers
    DENSE_LAYER_COUNT = 2   # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [15]  # the order for each layer
    MASK_BANDWIDTH = [20] # the consecutive features enabled at the input
    MASK_OVERLAP = [-5] # the overlap of observation between the a hidden and another
    EXTRA_FRAMES = 50  # the k extra frames beyond the middle frame (included by default)

    CLASS_NAMES = ['AC', 'CH' , 'CP', 'DB', 'Dr', 'EI' , 'GS' , 'Ja' , 'Si' ,'SM', 'Ra', 'Tr' ]