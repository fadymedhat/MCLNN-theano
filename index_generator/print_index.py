import os
import h5py

SRC_PATH = 'I:\ESC10-for-MCLNN\ESC10_folds_5_index'
FILE_PATH = 'Fold_02of05_ESC10_seedNoSeed_Validation.hdf5'

with h5py.File(os.path.join(SRC_PATH,FILE_PATH), "r") as hdf5_handle:
    print hdf5_handle[str('index')].value
