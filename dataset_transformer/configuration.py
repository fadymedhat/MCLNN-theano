

class ESC10:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 400
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'I:/dataset-esc10/ESC-10-masterstretched'
    DST_PATH = 'I:/dataset-esc10'

    DATASET_NAME = "esc10_COMMONPREPROCESS"
    # dataset standard file length = 5 seconds
    DEFAULT_DURATION = "5secs"
    # at a sampling rate of 22050 sample per second and nfft 1024 overlap 512 > 22050 * 5 sec / 512 = 215 frames
    FIRST_FRAME_IN_SLICE = 4  # to avoid disruptions at the beginning
    FRAME_NUM = 200  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 60
    FFT_BINS = 1024
    HOP_LENGTH_IN_SAMPLES = 512


    PROCESSING_BATCH = 10
    SLEEP_TIME = 2