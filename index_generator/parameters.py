
class ISMIR2004:
    DATASET = 'ismir2004'
    FOLD_COUNT = 10
    FOLDER_NAME = 'folds_indices_imnir'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0
    CLIP_COUNT_PER_CATEGORY_LIST = [640, 229, 52, 90, 203, 244]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

class ESC10:
    DATASET = 'esc10'
    FOLD_COUNT = 5
    FOLDER_NAME = 'folds_indices_esc10'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 0
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8

class ESC50:
    DATASET = 'esc50'
    FOLD_COUNT = 5
    FOLDER_NAME = 'folds_indices_esc50'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 0
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8


class BALLROOM:
    DATASET = 'ballroom'
    FOLD_COUNT = 10
    FOLDER_NAME = 'folds_indices_ballroom'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0
    CLIP_COUNT_PER_CATEGORY_LIST = []
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

class ESC10AUGMENTED:
    DATASET = 'esc10_12augment'
    FOLD_COUNT = 5
    FOLDER_NAME = 'folds_indices_esc10_aug'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 12
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8

class ESC50AUGMENTED:
    DATASET = 'esc50_4augment'
    FOLD_COUNT = 5
    FOLDER_NAME = 'folds_indices_esc50_aug'
    SHUFFLE_CATEGORY_CLIPS = False
    AUGMENTATION_VARIANTS_COUNT = 4
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                    40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8