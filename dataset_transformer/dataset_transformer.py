import os
import h5py
import time
import librosa
import numpy as np
from fnmatch import fnmatch

from configuration import ESC10

Config = ESC10


def store(file_handle, file_key, clip_list, sample_rate_list):

    for i, clip in enumerate(clip_list):
        print(i)
        file_key += 1
        # Passing through arguments to the Mel filters
        mel_spec = librosa.feature.melspectrogram(y=clip, sr=sample_rate_list[i], n_mels=Config.MEL_FILTERS_COUNT
                                              , n_fft=Config.FFT_BINS, hop_length=Config.HOP_LENGTH_IN_SAMPLES)



        # Convert to log scale (dB). We'll use the peak power as reference.
        log_mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)
        delta_log_mel_spec = librosa.feature.delta(log_mel_spec, width=9, order=1, axis=-1, trim=True)

        print('File: ' + str(file_key) + ' at SR:' + str(sr) + ' - Duration is :' + str(
            clip.shape[0] / sr) + ' sec - Spec size :' + str(mel_spec.shape))

        start = Config.FIRST_FRAME_IN_SLICE  # start of the segment
        end = start + Config.FRAME_NUM  # end of the segment

        if log_mel_spec.shape[1] < Config.FRAME_NUM:
            start = 0
            end = log_mel_spec.shape[1]
            print('SHORT included ' + str(y.shape[0] / 22050) + ' start :' + str(start) + ' <> end :' + str(end))


        log_mel_spec = np.transpose(log_mel_spec[:, start:end])
        delta_log_mel_spec = np.transpose(delta_log_mel_spec[:, start:end])
        spec_delta = np.concatenate((log_mel_spec, delta_log_mel_spec), axis=1)

        file_handle.create_dataset(str(file_key), (spec_delta.shape[0], spec_delta.shape[1]), data=spec_delta,
                         dtype='float32')



    return file_key


if __name__ == '__main__':
    # load n fils
        # transform n files
        # store n files

    file_key = -1
    batch_counter = 0
    clip_list = []
    sample_rate_list = []

    clip_name_txt_handle = open(Config.DATASET_NAME+".txt","w")
    # initialize hdf5 file
    hdf5_handle = h5py.File(os.path.join(Config.DST_PATH, Config.DATASET_NAME + "Specmeln_mels="
                               + str(Config.MEL_FILTERS_COUNT) + "_nfft=" + str(Config.FFT_BINS) + "_hoplength="
                               + str(Config.HOP_LENGTH_IN_SAMPLES) + "_fmax=NIL_22050hzsampling_FF="
                               + str(Config.FIRST_FRAME_IN_SLICE) + "_FN=" + str(Config.FRAME_NUM)
                               + "_" + Config.DEFAULT_DURATION + "Delta.hdf5"), "w")


    counter =  -1
    # start walking through the files
    for path, subdirs, files in sorted(os.walk(Config.SRC_PATH, topdown=False)):
        for name in sorted(files):
            if fnmatch(name, "*.wav"):
                counter +=1
                clip_path = os.path.join(path, name)
                clip_name_txt_handle.write(clip_path + '\n')



                y, sr = librosa.load(clip_path, sr=22050, mono=True)

                clip_list.append(y)
                sample_rate_list.append(sr)
                batch_counter += 1

                if batch_counter % Config.PROCESSING_BATCH == 0:
                    file_key = store(hdf5_handle, file_key, clip_list, sample_rate_list)
                    batch_counter = 0
                    clip_list = []
                    sample_rate_list = []
                    time.sleep(Config.SLEEP_TIME)

    if batch_counter != 0:
        file_key = store(hdf5_handle, file_key, clip_list, sample_rate_list)

    clip_name_txt_handle.close()
    print('Total files :' + str(file_key+1) + ' out of ' + str(Config.DATASET_ORIGINAL_FILE_COUNT)
          + ' increment concat count in the previous processing stage if there is a mismatch')
    print('Counter',counter+1)