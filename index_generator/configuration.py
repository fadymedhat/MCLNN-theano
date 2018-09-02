import os


class ESC10:
    DATASET = 'esc10'
    DST_PATH = 'I:/dataset-esc10_'
    FOLD_COUNT = 5
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 0
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8


class ESC10AUGMENTED:
    DATASET = 'esc10_12augmentations'
    DST_PATH = 'I:/dataset-esc10AUG'
    FOLD_COUNT = 5
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 12
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8


class ESC50:
    DATASET = 'esc50'
    DST_PATH = 'I:/dataset-esc50GIT'
    FOLD_COUNT = 5
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 0
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8


class ESC50AUGMENTED:
    DATASET = 'esc50_4augmentations'
    DST_PATH = 'I:/dataset-esc50AUG'
    FOLD_COUNT = 5
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 4
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8


class URBANSOUND8K:
    DATASET = 'UrbanSound8K'
    DST_PATH = 'I:/dataset-UrbanSound8K'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 0

    # samples per category following the category order: ('AC', 'CH', 'CP', 'DB', 'Dr', 'EI', 'GS', 'Ja', 'Si', 'SM')
    CLIP_COUNT_PER_CATEGORY_LIST = [1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

    CSV_FILE_PATH = os.path.join(DST_PATH, 'UrbanSound8KwithFileSeq.csv')
    COL_FILE_SEQ = 0 # csv column index for file sequence - file sequence zero indexed
    COL_FOLD_ID = 7 # csv column index for fold id of a file - fold id is 1 indexed
    COL_CLASS_ID = 8 # csv column index for class id of a file - class id is zero indexed

class YORNOISE:
    DATASET = 'YorNoise'
    DST_PATH = 'I:/dataset-YorNoiseGIT'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 0

    # samples per category following the category order: ('AC', 'CH', 'CP', 'DB', 'Dr', 'EI', 'GS', 'Ja', 'Si', 'SM', 'Ra', 'Tr')
    CLIP_COUNT_PER_CATEGORY_LIST = [1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000, 620, 907]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

    CSV_FILE_PATH = os.path.join(DST_PATH, 'UrbanSound8KwithFileSeqYork.csv')
    COL_FILE_SEQ = 0 # csv column index for file sequence - file sequence zero indexed
    COL_FOLD_ID = 7 # csv column index for fold id of a file - fold id is 1 indexed
    COL_CLASS_ID = 8 # csv column index for class id of a file - class id is zero indexed

class HOMBURG:
    DATASET = 'homburg'
    DST_PATH = 'I:/dataset-homburgGIT'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0

    # samples per category following the category order: ('Al', 'Bl', 'El', 'FC', 'FS', 'Ja', 'Po', 'RH', 'Ro')
    CLIP_COUNT_PER_CATEGORY_LIST = [145, 120, 113, 222, 47, 319, 116, 300, 504]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

class GTZAN:
    DATASET = 'gtzan'
    DST_PATH = 'I:/dataset-gtzanGIT'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0

    # samples per category following the category order: ('Bl', 'Cl', 'Co', 'Di', 'Hi', 'Ja', 'Me', 'Po', 'Re', 'Ro')
    CLIP_COUNT_PER_CATEGORY_LIST = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

class ISMIR2004:
    DATASET = 'ismir2004'
    DST_PATH = 'I:/dataset-ismir2004GIT'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_indices'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0

    # samples per category following the category order: ('Cl', 'El', 'Ja', 'Me', 'Po', 'Wo')
    CLIP_COUNT_PER_CATEGORY_LIST = [640, 229, 52, 90, 203, 244]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1


class BALLROOM:
    DATASET = 'ballroom'
    DST_PATH = 'I:/dataset-ballroomGIT'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_indices'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0
    # samples per category following the category order: ('CC', 'Ji', 'QS', 'Ru', 'Sa', 'Ta', 'VW', 'Wa')
    CLIP_COUNT_PER_CATEGORY_LIST = [111, 60, 82, 98, 86, 86, 65, 110]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1
