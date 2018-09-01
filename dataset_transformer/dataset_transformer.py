import os
import h5py
import time
import librosa
import numpy as np
from fnmatch import fnmatch

from configuration import ESC10, ESC10AUGMENTED, ESC50, ESC50AUGMENTED, URBANSOUND8K, YORNOISE

Config = YORNOISE


def store(file_handle, file_key, clip_list, sample_rate_list, short_count):
    for i, clip in enumerate(clip_list):
        print(i)
        file_key += 1
        sr = sample_rate_list[i]
        # Passing through arguments to the Mel filters
        mel_spec = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=Config.MEL_FILTERS_COUNT
                                                  , n_fft=Config.FFT_BINS, hop_length=Config.HOP_LENGTH_IN_SAMPLES)

        # Convert to log scale (dB), peak power is a reference.
        log_mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)

        if Config.INCLUDE_DELTA == True:
            delta_log_mel_spec = librosa.feature.delta(log_mel_spec, width=9, order=1, axis=-1, trim=True)

        print('File: ' + str(file_key) + ' at SR:' + str(sr) + ' - Duration is :' + str(
            clip.shape[0] / sr) + ' sec - Spec size :' + str(mel_spec.shape))

        start = Config.FIRST_FRAME_IN_SLICE  # start of the segment
        end = start + Config.FRAME_NUM  # end of the segment

        if log_mel_spec.shape[1] < Config.FRAME_NUM:
            start = 0
            end = log_mel_spec.shape[1]
            print('SHORT included ' + str(clip.shape[0] / sr) + ' start :' + str(start) + ' <> end :' + str(end))
            short_count += 1

        spectrogram = np.transpose(log_mel_spec[:, start:end])

        if Config.INCLUDE_DELTA == True:
            spectrogram_delta = np.transpose(delta_log_mel_spec[:, start:end])
            spectrogram = np.concatenate((spectrogram, spectrogram_delta), axis=1)

        file_handle.create_dataset(str(file_key), (spectrogram.shape[0], spectrogram.shape[1]), data=spectrogram,
                                   dtype='float32')

    return file_key, short_count


def load(clip_path_list):

    clip_list = []
    sample_rate_list = []
    for i in range(len(clip_path_list)):
        y, sr = librosa.load(clip_path_list[i], sr=22050, mono=True)

        clip_list.append(y)
        sample_rate_list.append(sr)

    return clip_list, sample_rate_list


def process_batch(clip_path_list, file_key, short_count):

    clip_list, sample_rate_list = load(clip_path_list)
    file_key, short_count = store(hdf5_handle, file_key, clip_list, sample_rate_list, short_count)
    time.sleep(Config.SLEEP_TIME)
    return file_key, short_count

def navigate_directory(clip_name_txt_handle):
    counter = -1
    file_key = -1
    short_count = 0
    clip_path_list = []
    # start walking through the files
    for path, subdirs, files in sorted(os.walk(Config.SRC_PATH, topdown=False)):
        for name in sorted(files):
            if fnmatch(name, "*.wav"):
                counter += 1
                clip_path = os.path.join(path, name)
                clip_name_txt_handle.write(clip_path + '\n')
                clip_path_list.append(clip_path)
                if len(clip_path_list) == Config.PROCESSING_BATCH:
                    file_key, short_count = process_batch(clip_path_list, file_key, short_count)
                    clip_path_list = []

    if len(clip_path_list) != 0:
        file_key, short_count = process_batch(clip_path_list, file_key, short_count)

    return file_key, short_count, counter

def navigate_csv(clip_name_txt_handle):
    counter = -1
    file_key = -1
    short_count = 0
    clip_path_list = []
    with open(Config.CSV_FILE_PATH) as csvfile:

        csvfile.readline()  # skip first row
        for row in csvfile:
            fields = row.strip().split(",")  # make into fields

            class_folder = fields[Config.COL_FOLDER_NAME]
            file_name = fields[Config.COL_FILE_NAME]
            clip_path = os.path.join(Config.SRC_PATH, class_folder, file_name)
            if fnmatch(clip_path, "*.wav"):
                counter += 1
                clip_name_txt_handle.write(clip_path + '\n')
                clip_path_list.append(clip_path)
                if len(clip_path_list) == Config.PROCESSING_BATCH:
                    file_key, short_count = process_batch(clip_path_list, file_key, short_count)
                    clip_path_list = []


        if len(clip_path_list) != 0:
            file_key, short_count = process_batch(clip_path_list, file_key, short_count)

    return file_key, short_count, counter


if __name__ == '__main__':
    # load n fils
    # transform n files
    # store n files


    # clip_list = []
    # sample_rate_list = []

    if not os.path.exists(Config.DST_PATH):
        os.makedirs(Config.DST_PATH)

    full_dataset_filename = Config.DATASET_NAME + "Specmeln_mels=" + str(Config.MEL_FILTERS_COUNT) + "_nfft=" + str(
        Config.FFT_BINS) + "_hoplength=" + str(Config.HOP_LENGTH_IN_SAMPLES) + "_fmax=NIL_22050hzsampling_FF=" + str(
        Config.FIRST_FRAME_IN_SLICE) + "_FN=" + str(Config.FRAME_NUM) + "_" + Config.DEFAULT_DURATION

    if Config.INCLUDE_DELTA == True:
        full_dataset_filename += 'Delta'

    clip_name_txt_handle = open(os.path.join(Config.DST_PATH, full_dataset_filename + "_storage_ordering.txt"), "w")
    # initialize hdf5 file
    hdf5_handle = h5py.File(os.path.join(Config.DST_PATH, full_dataset_filename + ".hdf5"), "w")

    if hasattr(Config, 'CSV_FILE_PATH') == True:
        file_key, short_count, counter = navigate_csv(clip_name_txt_handle)
    else:
        file_key, short_count, counter = navigate_directory(clip_name_txt_handle)



    clip_name_txt_handle.close()


    print('Total files :' + str(file_key + 1) + ' out of ' + str(Config.DATASET_ORIGINAL_FILE_COUNT)
          + ' increment concat count in the previous processing stage if there is a mismatch')
    print('Counter', counter + 1)
    print('short_count', short_count)
