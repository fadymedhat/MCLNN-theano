import os
import h5py
import numpy as np

from configuration import ESC10, ESC10AUGMENTED, ESC50, ESC50AUGMENTED, URBANSOUND8K, YORNOISE, BALLROOM, ISMIR2004

from numpy import random as rand

# random.seed(None)
# random.seed(1754157958)
# random.seed(0)

Config = YORNOISE

rand.seed(1754157958)

seed = np.random.get_state()[1][0]

if Config.SHUFFLE_CATEGORY_CLIPS == False:
    seed = 'NoSeed'


class IndexGenerator():
    def store_fold(self, fold_type, index, label, fold_id):

        path_name = os.path.join(Config.DST_PATH, Config.FOLDER_NAME)
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        hdf5_handle = h5py.File(os.path.join(path_name,
                                             'Fold_' + str(fold_id + 1).zfill(2) + 'of' + str(Config.FOLD_COUNT).zfill(
                                                 2) + '_' + Config.DATASET.upper() + '_seed' + str(
                                                 seed) + '_' + fold_type.title() + '.hdf5'), "w")

        hdf5_handle.create_dataset(str('index'), data=index,
                                   dtype='int32')

        hdf5_handle.create_dataset(str('label'), data=label,
                                   dtype='int32')

    def load_fold(self, fold_type, fold_id):
        """

        :param fold_type:
        :param fold_id:
        :return:
        """

        with h5py.File(os.path.join(Config.DST_PATH, Config.FOLDER_NAME,
                                    'Fold_' + str(fold_id + 1).zfill(2) + 'of' + str(Config.FOLD_COUNT).zfill(2)
                                            + '_' + Config.DATASET.upper()
                                            + '_seed' + str(seed)
                                            + '_' + fold_type.title() + '.hdf5'), "r") as hdf5_handle:
            print fold_type, 'INDICES of fold :', fold_id, 'are', hdf5_handle[str('index')].value
            print fold_type, 'LABELS of fold :', fold_id, 'are', hdf5_handle[str('label')].value

    def generate_consecutive_index_all_categories(self, shuffle):
        """

        :return:
        """
        offset = 0  # offset to consider previous class elements count in current one
        clip_index_list = []
        clip_label_list = []
        temp_index_list = []

        for k in range(class_count):
            clip_count_per_category = clip_count_per_category_list[k]

            if shuffle == True:
                temp_index_list = rand.permutation(clip_count_per_category)
            else:
                temp_index_list = np.arange(clip_count_per_category)

            clip_index_list.append(temp_index_list + offset)
            clip_label_list.append([k] * clip_count_per_category)
            offset += clip_count_per_category

        return clip_index_list, clip_label_list

    def assign_indices_to_folds(self, clip_index_list, clip_label_list):
        """

        :param clip_index_list:
        :param clip_label_list:
        :return:
        """

        clip_index_vector = np.concatenate(clip_index_list)
        clip_label_vector = np.concatenate(clip_label_list)

        assigned_indices_count = 0
        fold_cells = [[] for _ in range(Config.FOLD_COUNT)]
        fold_cells_label = [[] for _ in range(Config.FOLD_COUNT)]

        while assigned_indices_count < len(clip_index_vector):
            for fold_id in range(Config.FOLD_COUNT):
                if assigned_indices_count < len(clip_index_vector):
                    start_index = assigned_indices_count
                    last_index = assigned_indices_count + batch_size_per_fold_assignment
                    fold_cells[fold_id] += clip_index_vector[start_index:last_index].tolist()
                    fold_cells_label[fold_id] += clip_label_vector[start_index:last_index].tolist()
                    assigned_indices_count = last_index

        fold_cells = np.asarray(fold_cells)
        fold_cells_label = np.asarray(fold_cells_label)

        return fold_cells, fold_cells_label

    def double_check_index_assignment(self, fold_cells_label):
        """

        :param fold_cells_label:
        :return:
        """
        hist_sum = []
        for n in range(Config.FOLD_COUNT):
            hist_row = np.histogram(fold_cells_label[n], len(clip_count_per_category_list))[0]
            print hist_row, 'sum per fold =', np.sum(hist_row)
            hist_sum.append(hist_row)
            # histc(fold_cells_label[n],[1:10])

        np_hist_sum = np.asarray(hist_sum)

        print np.sum(np_hist_sum, axis=0), np.sum(
            np.sum(np_hist_sum, axis=0)), 'should match the original dataset clip-count per category'

    def load_index_from_csv(self, file_path):

        fold_cells = [[] for _ in range(Config.FOLD_COUNT)]
        fold_cells_label = [[] for _ in range(Config.FOLD_COUNT)]

        with open(file_path) as csvfile:
            csvfile.readline()  # skip first row
            for row in csvfile:
                fields = row.strip().split(",")  # make Into fields
                # print ('Fold ID', int(fields[Config.COL_FOLD_ID]) - 1, ' ', [int(fields[Config.COL_FILE_SEQ])])
                fold_cells[int(fields[Config.COL_FOLD_ID]) - 1] += [int(fields[Config.COL_FILE_SEQ])]
                fold_cells_label[int(fields[Config.COL_FOLD_ID]) - 1] += [int(fields[Config.COL_CLASS_ID])]

        fold_cells = np.asarray(fold_cells)
        fold_cells_label = np.asarray(fold_cells_label)

        return fold_cells, fold_cells_label

    def generate_data_split(self, fold_cells, fold_cells_label):
        """

        :param fold_cells:
        :param fold_cells_label:
        :return:
        """

        test_set_index = []
        validation_set_index = []
        train_set_index = []

        test_set_label = []
        validation_set_label = []
        train_set_label = []

        fold_index_list = np.arange(Config.FOLD_COUNT)

        for fold_id in range(Config.FOLD_COUNT):

            test_set_index = fold_cells[fold_index_list[0]][0::(Config.AUGMENTATION_VARIANTS_COUNT + 1)]
            validation_set_index = fold_cells[fold_index_list[1]][0::(Config.AUGMENTATION_VARIANTS_COUNT + 1)]
            train_set_index = np.concatenate(fold_cells[fold_index_list[2::]])

            test_set_label = fold_cells_label[fold_index_list[0]][0::(Config.AUGMENTATION_VARIANTS_COUNT + 1)]
            validation_set_label = fold_cells_label[fold_index_list[1]][0::(Config.AUGMENTATION_VARIANTS_COUNT + 1)]
            train_set_label = np.concatenate(fold_cells_label[fold_index_list[2::]])

            # _______ End of Text :) _______________

            print('Test no. : ', len(test_set_index));
            print('Validation no. : ', len(validation_set_index));
            print('Train no. : ', len(train_set_index));
            print('            Total no. : ', len(test_set_index) + len(validation_set_index) + len(train_set_index));

            inter1 = len(np.intersect1d(test_set_index, validation_set_index));
            inter2 = len(np.intersect1d(test_set_index, train_set_index));
            inter3 = len(np.intersect1d(validation_set_index, train_set_index));

            if (inter1 + inter2 + inter3) > 0:
                print(' +++++++++++++++++++++++ \n')
                print(' Possible intersection between dataset indices  \n')
                print(' +++++++++++++++++++++++ \n')
                quit()

            self.store_fold('test', test_set_index, test_set_label, fold_id)
            self.store_fold('validation', validation_set_index, validation_set_label, fold_id)
            self.store_fold('train', train_set_index, train_set_label, fold_id)

            fold_index_list = np.roll(fold_index_list, 1);


if __name__ == '__main__':

    class_count = len(Config.CLIP_COUNT_PER_CATEGORY_LIST);
    clip_count_per_category_list = np.asarray(Config.CLIP_COUNT_PER_CATEGORY_LIST) * (
        Config.AUGMENTATION_VARIANTS_COUNT + 1)
    batch_size_per_fold_assignment = Config.BATCH_SIZE_PER_FOLD_ASSIGNMENT * (Config.AUGMENTATION_VARIANTS_COUNT + 1)

    index_generator = IndexGenerator()

    if hasattr(Config, 'CSV_FILE_PATH') == True:
        fold_cells, fold_cells_label = index_generator.load_index_from_csv(Config.CSV_FILE_PATH)
    else:
        clip_index_list, clip_label_list = index_generator.generate_consecutive_index_all_categories(
            Config.SHUFFLE_CATEGORY_CLIPS)
        fold_cells, fold_cells_label = index_generator.assign_indices_to_folds(clip_index_list, clip_label_list)

    index_generator.double_check_index_assignment(fold_cells_label)
    index_generator.generate_data_split(fold_cells, fold_cells_label)

    # have a look inside for debugging
    for i in range(Config.FOLD_COUNT):
        index_generator.load_fold('test', i)
        # index_generator.load_fold('validation', i)
        # index_generator.load_fold('train', i)
