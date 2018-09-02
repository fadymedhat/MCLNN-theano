import os

class ESC10:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 400
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'I:/dataset-esc10/ESC-10-masterstretched'
    DST_PATH = 'I:/dataset-esc10_'

    DATASET_NAME = "esc10"
    # dataset standard file length = 5 seconds
    DEFAULT_DURATION = "5secs"
    # at a sampling rate of 22050 sample per second and nfft 1024 overlap 512 > 22050 * 5 sec / 512 = 215 frames
    FIRST_FRAME_IN_SLICE = 4  # to avoid disruptions at the beginning
    FRAME_NUM = 200  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 60
    FFT_BINS = 1024
    HOP_LENGTH_IN_SAMPLES = 512

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = True

class ESC10AUGMENTED:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 12
    DATASET_ORIGINAL_FILE_COUNT = 400
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'I:/dataset-esc10/ESC-10-JAMS_12variants'
    DST_PATH = 'I:/dataset-esc10AUG'

    DATASET_NAME = "esc10_AUG"
    # dataset standard file length = 5 seconds
    DEFAULT_DURATION = "5secs"
    # at a sampling rate of 22050 sample per second and nfft 1024 overlap 512 > 22050 * 5 sec / 512 = 215 frames
    FIRST_FRAME_IN_SLICE = 4  # to avoid disruptions at the beginning
    FRAME_NUM = 200  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 60
    FFT_BINS = 1024
    HOP_LENGTH_IN_SAMPLES = 512

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = True

class ESC50:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 2000
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'I:/dataset-esc50/ESC-50-masterstretched'
    DST_PATH = 'I:/dataset-esc50GIT'

    DATASET_NAME = "esc50"
    # dataset standard file length = 5 seconds
    DEFAULT_DURATION = "5secs"
    # at a sampling rate of 22050 sample per second and nfft 1024 overlap 512 > 22050 * 5 sec / 512 = 215 frames
    FIRST_FRAME_IN_SLICE = 4  # to avoid disruptions at the beginning
    FRAME_NUM = 200  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 60
    FFT_BINS = 1024
    HOP_LENGTH_IN_SAMPLES = 512

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = True

class ESC50AUGMENTED:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 4
    DATASET_ORIGINAL_FILE_COUNT = 2000
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'I:/dataset-esc50/ESC-50-JAMS'
    DST_PATH = 'I:/dataset-esc50AUG'

    DATASET_NAME = "esc50_4augmentations"
    # dataset standard file length = 5 seconds
    DEFAULT_DURATION = "5secs"
    # at a sampling rate of 22050 sample per second and nfft 1024 overlap 512 > 22050 * 5 sec / 512 = 215 frames
    FIRST_FRAME_IN_SLICE = 4  # to avoid disruptions at the beginning
    FRAME_NUM = 200  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 60
    FFT_BINS = 1024
    HOP_LENGTH_IN_SAMPLES = 512

    PROCESSING_BATCH = 10
    SLEEP_TIME = 1
    INCLUDE_DELTA = True

class URBANSOUND8K:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 8732
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'I:/dataset-urbansound8k/UrbanSound8KexpandedSmallfiles/audiostretch'
    DST_PATH = 'I:/dataset-UrbanDST'

    DATASET_NAME = "urbansound8k"
    # dataset standard file length = 4 seconds
    DEFAULT_DURATION = "4secs"
    # at a sampling rate of 22050 sample per second and nfft 1024 overlap 512 > 22050 * 4 sec / 512 = 172 frames
    FIRST_FRAME_IN_SLICE = 1  # to avoid disruptions at the beginning
    FRAME_NUM = 170  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 60
    FFT_BINS = 1024
    HOP_LENGTH_IN_SAMPLES = 512

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = True

    CSV_FILE_PATH = os.path.join(DST_PATH, 'UrbanSound8KwithFileSeq.csv')
    COL_FILE_NAME = 2 # csv column index for file name
    COL_FOLDER_NAME = 9 # csv column index for folder name (class category)

class YORNOISE:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 10259
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'I:/dataset-urbansound8k/UrbanSound8KexpandedSmallfiles/audiostretch'
    DST_PATH = 'I:/dataset-YorNoiseGIT'

    DATASET_NAME = "yornoise"
    # dataset standard file length = 4 seconds
    DEFAULT_DURATION = "4secs"
    # at a sampling rate of 22050 sample per second and nfft 1024 overlap 512 > 22050 * 4 sec / 512 = 172 frames
    FIRST_FRAME_IN_SLICE = 1  # to avoid disruptions at the beginning
    FRAME_NUM = 170  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 60
    FFT_BINS = 1024
    HOP_LENGTH_IN_SAMPLES = 512

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = True

    CSV_FILE_PATH = os.path.join(DST_PATH, 'UrbanSound8KwithFileSeqYork.csv')
    COL_FILE_NAME = 2 # csv column index for file name
    COL_FOLDER_NAME = 9 # csv column index for folder name (class category)

class HOMBURG:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 1886
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'F:/dataset-homburg/homburg-mp3towav'
    DST_PATH = 'I:/dataset-homburgGIT'

    DATASET_NAME = "homburg"
    # dataset standard file length of the Homburg = 10 seconds
    DEFAULT_DURATION = "10secs"
    # at a sampling rate of 22050 sample per second and nfft 2048 overlap 1024 > 22050 * 10 sec / 1024 = 215 frames
    FIRST_FRAME_IN_SLICE = 8  # to avoid disruptions at the beginning
    FRAME_NUM = 200  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 256
    FFT_BINS = 2048
    HOP_LENGTH_IN_SAMPLES = 1024

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = False

class GTZAN:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 1000
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'G:/dataset-gtzan/gtzan-autowav'
    DST_PATH = 'I:/dataset-gtzanGIT'

    DATASET_NAME = "gtzan"
    # dataset standard file length of the GTZAN = 30 seconds
    DEFAULT_DURATION = "30secs"
    # at a sampling rate of 22050 sample per second and nfft 2048 overlap 1024 > 22050 * 30 sec / 1024 = 645 frames
    FIRST_FRAME_IN_SLICE = 23  # to avoid disruptions at the beginning
    FRAME_NUM = 600  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 256
    FFT_BINS = 2048
    HOP_LENGTH_IN_SAMPLES = 1024

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = False

class ISMIR2004:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 1458
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'G:\dataset-ISMIR2004\ismir_combined'
    DST_PATH = 'I:\dataset-ISMIR2004GIT'

    DATASET_NAME = "ismir2004"
    # different length for each file in the ISMIR2004. we will extract the second 30secs from each file
    DEFAULT_DURATION = "30secs"
    # at a sampling rate of 22050 sample per second and nfft 2048 overlap 1024 > 22050 * 30 sec / 1024 = 645 frames
    FIRST_FRAME_IN_SLICE = 600  # to avoid disruptions at the beginning
    FRAME_NUM = 600  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 256
    FFT_BINS = 2048
    HOP_LENGTH_IN_SAMPLES = 1024

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = False

class BALLROOM:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 698
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = 'F:/dataset-ballroom/BallroomData'
    DST_PATH = 'I:/dataset-ballroomGIT'

    DATASET_NAME = "ballroom"
    # dataset standard file length of the Ballroom = 30 seconds
    DEFAULT_DURATION = "30secs"
    # at a sampling rate of 22050 sample per second and nfft 2048 overlap 1024 > 22050 * 30 sec / 1024 = 645 frames
    FIRST_FRAME_IN_SLICE = 23  # to avoid disruptions at the beginning
    FRAME_NUM = 600  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 256
    FFT_BINS = 2048
    HOP_LENGTH_IN_SAMPLES = 1024

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0
    INCLUDE_DELTA = False