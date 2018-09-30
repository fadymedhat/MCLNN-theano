import os
import numpy as np
import matplotlib.pyplot as pyplt
from sklearn import preprocessing
from keras.models import Model


class Visualizer(object):
    def __init__(self, configuration):

        self.plt = pyplt
        self.plt.ioff()
        self.plt.axis('off')
        self.plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            left='off',
            right='off',
            labelbottom='off',
            labelleft='off')

        self.visualization_parent_path = configuration.VISUALIZATION_PARENT_PATH
        self.image_count = configuration.SAVE_TEST_SEGMENT_PREDICTION_IMAGE_COUNT
        self.initial_segment_index = configuration.SAVE_TEST_SEGMENT_PREDICTION_INITIAL_SEGMENT_INDEX
        self.hidden_nodes_slices_count = configuration.HIDDEN_NODES_SLICES_COUNT

    def save_image(self, image_matrix, image_name, image_path, colormap='gray', transpose=True,
                   normalize_per_feature=True, interpolate=False):
        folder_path = os.path.join(image_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        absolute_path = os.path.join(folder_path, image_name + '.png')

        scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))

        if normalize_per_feature == True:
            image_matrix = scaler.fit_transform(image_matrix)  # .reshape(image_matrix.shape[0], -1).transpose()
        elif normalize_per_feature == False:
            scaled_segment = scaler.fit_transform(image_matrix.reshape(-1, 1))
            image_matrix = scaled_segment.reshape(image_matrix.shape[0], -1)

        if transpose == True:
            image_matrix = np.transpose(image_matrix)

        if interpolate == True:
            interpolate = None
        else:
            interpolate = 'none'

        self.plt.imshow(image_matrix,
                        # vmin=0, vmax=1,
                        interpolation=interpolate,
                        cmap=colormap,
                        aspect='equal')
        self.plt.savefig(absolute_path, dpi=300,
                         bbox_inches='tight', pad_inches=0)

    def visualize_model_weights(self, model, path):
        #HIDDEN_NODES_SLICES_COUNT = 40
        for i in range(len(model.layers)):

            layer_name = model.layers[i].name
            layer_type = layer_name.strip('0123456789')
            if layer_type == 'flatter':
                return

            if layer_type in ['mclnn', 'clnn']:
                layer_weights = model.layers[i].W.get_value(borrow=True)
                weight_transposed = layer_weights.transpose(2, 1, 0)
                weight_chunk = weight_transposed[0:self.hidden_nodes_slices_count, :, :]
                weight_slices = np.reshape(weight_chunk.transpose(1, 0, 2), (weight_chunk.shape[1], -1))
                self.save_image(image_matrix=weight_slices, image_name='weights_'+layer_type+'_layer_' + str(i),
                                image_path=path, colormap='gray',
                                transpose=False, normalize_per_feature=False,interpolate=True)

    def visualize_input_segments(self, segments, path, initial_segment):

        for i in range(len(segments)):
            self.save_image(image_matrix=segments[i],
                            image_name='input__segment__' + str(initial_segment + i),
                            image_path=path,
                            colormap='jet')

    def visualize_prediction_segments(self, model, segments, path, initial_segment, epoch_id='',
                                      layer_filter_list=['prelu', 'mclnn', 'clnn'], first_mclnn_only=False):


        for i in range(len(model.layers)):

            layer_name = model.layers[i].name
            layer_type = layer_name.strip('0123456789')
            if layer_type == 'flatter':
                return

            if layer_type in layer_filter_list:
                intermediate_model = Model(input=model.input,
                                           output=model.get_layer(layer_name).output)
                intermediate_prediction = intermediate_model.predict(
                    segments, batch_size=1)
                perdicted_segments_path = os.path.join(path, layer_name)
                for j in range(len(intermediate_prediction)):
                    self.save_image(image_matrix=intermediate_prediction[j], image_path=perdicted_segments_path,
                                    image_name='segment_' + str(j + initial_segment) + '__' + str(epoch_id))
                if first_mclnn_only == True:
                    break

    def visualize_weights_and_sample_test_clip(self, model, data_loader):

        # weights visualization
        self.visualize_model_weights(model=model, path=self.visualization_parent_path)

        # input segments visualization
        input_segments_path = os.path.join(self.visualization_parent_path, 'input_segment')
        self.visualize_input_segments(
            segments=data_loader.test_segments[self.initial_segment_index:self.initial_segment_index + self.image_count],
            path=input_segments_path, initial_segment=self.initial_segment_index)

        # predicted segments visualization
        predicted_segments_path = os.path.join(self.visualization_parent_path, 'predicted_segment')
        self.visualize_prediction_segments(model=model,
                                           segments=data_loader.test_segments[
                                                    self.initial_segment_index:self.initial_segment_index + self.image_count],
                                           path=predicted_segments_path, initial_segment=self.initial_segment_index)
