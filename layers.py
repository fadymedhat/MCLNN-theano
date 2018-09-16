# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import theano
from theano import tensor as T, printing
from keras import activations, initializations, regularizers, constraints
from keras import backend as K
from keras.backend.common import _FLOATX
from keras.engine import InputSpec, Layer

# INTERNAL UTILS
theano.config.floatX = _FLOATX
#_LEARNING_PHASE = T.scalar(dtype='uint8', name='keras_learning_phase')  # 0 = test, 1 = train


class MaskedConditional(Layer):
    '''Masked Conditional Neural Network layer.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see Keras [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see Keras [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        order: frames in a single temporal direction of an MCLNN
        bandwidth: consecutive 1's to enable features in a single feature vector.
        overlap: overlapping distance between two neighbouring hidden nodes.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        3D tensor with shape:  (nb_samples, input_row,input_dim)`.
    # Output shape
        3D tensor with shape:  (nb_samples, output_row,output_dim)`.
    '''

    def __init__(self, output_dim, init='glorot_uniform', activation='linear',
                 weights=None, order=None, bandwidth=None, overlap=None, layer_is_masked=True,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.order = order
        self.bandwidth = bandwidth
        self.overlap = overlap
        self.layer_is_masked=layer_is_masked

        # --K_START -- Refer to keras documentation for the below parameters.
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        # --K_END -- Refer to keras documentation for the above parameters.

        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaskedConditional, self).__init__(**kwargs)

    def printme(self, name, mat):
        return printing.Print('vector')(mat)

    def printdim(self, name, mat):
        return printing.Print(name, attrs=['shape'])(mat)

    def construct_mask_original(self, feature_count, hidden_count, bandwidth, overlap):

        binary_mask = np.zeros([hidden_count, feature_count])
        base_index = np.arange(0, feature_count * hidden_count, feature_count + (bandwidth - overlap))
        base_location = np.tile(base_index, bandwidth)
        base_location = base_location.reshape([bandwidth, base_index.shape[0]])
        base_location = base_location.transpose()

        increment = np.arange(0, bandwidth)
        increment = increment.transpose()
        increment = np.tile(increment, base_index.shape[0])
        increment = increment.reshape([base_index.shape[0], bandwidth])
        linear_index = base_location + increment

        # make sure the generated indices reside within the size of the mask
        linear_index_filtered = linear_index.astype(int).flatten()
        linear_index_filtered = linear_index_filtered[linear_index_filtered < np.prod(binary_mask.shape)]

        # fill in the mask with the filter-like pattern of binary 1 values
        binary_mask.flat[linear_index_filtered] = 1
        binary_mask = binary_mask.transpose()
        binary_mask = np.asarray(binary_mask, np.float32)
        # print('=============================== DISABLE MASK  (CLNN)===============================')
        # binary_mask = np.ones(binary_mask.shape, np.float32)
        # print('Layer featcount: ' + str(feature_count) + ' bandwidth: ' + str(bandwidth) + ' overlap: ' + str(
        #     overlap) + ' maskcol: ' + str( hidden_count))
        return binary_mask

    def construct_mask(self, feature_count, hidden_count, bandwidth, overlap, layer_is_masked):

        bw = bandwidth
        ov = overlap
        l = feature_count
        e = hidden_count

        a = np.arange(1, bw + 1)
        g = np.arange(1, int(np.ceil((l * e) / (l + bw - ov))) + 1)

        if layer_is_masked is False:
            binary_mask = np.ones([l, e])
        else:
            mask = np.zeros([l, e])
            flat_matrix = mask.flatten('F')

            for i in range(len(a)):
                for j in range(len(g)):
                    lx = a[i] + (g[j] - 1) * (l + bw - ov)
                    if lx <= l * e:
                        flat_matrix[lx - 1] = 1

            binary_mask =  np.transpose(flat_matrix.reshape(e, l))

        # binary_mask = np.ones(binary_mask.shape, np.float32)
        # print('Layer featcount: ' + str(feature_count) + ' bandwidth: ' + str(bandwidth) + ' overlap: ' + str(
        #     overlap) + ' maskcol: ' + str( hidden_count))

        return binary_mask.astype(np.float32)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_shape[1], input_dim))]

        self.W = self.init((self.order * 2 + 1, input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        # self.W = self.print_dim('W :',self.W)
        self.b = K.zeros((self.output_dim,),
                         name='{}_b'.format(self.name))

        self.trainable_weights = [self.W, self.b]

        self.weightmask = self.construct_mask(feature_count=input_dim,
                                              hidden_count=self.output_dim,
                                              bandwidth=self.bandwidth,
                                              overlap=self.overlap,
                                              layer_is_masked=self.layer_is_masked)

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):

        mini_batch = x
        segment_count = mini_batch.shape[0]  # number of samples in a minibatch
        segment_length = mini_batch.shape[1]
        feature_count = mini_batch.shape[2]

        concatenated_segments = mini_batch.reshape((segment_count * segment_length, feature_count))
        # concatenated_segments = self.print_dim('concatenated_segments',concatenated_segments)

        frame_count_per_minibatch = concatenated_segments.shape[
            0]  # number of frames after concatenating the minibatch samples
        # frame_count_per_minibatch = self.print_me('frame_count_per_minibatch', frame_count_per_minibatch)
        frames_index_per_minibatch = T.arange((frame_count_per_minibatch))  # index vector for all the frames
        ##frames_index_per_minibatch = self.print_dim('frames_index_per_minibatch', frames_index_per_minibatch)


        # reshaping the index vector to minibatch * segment_length
        frames_index_per_segment_matrix = frames_index_per_minibatch.reshape((segment_count, segment_length))

        # frames_index_per_segment_matrix = self.print_dim('frames_index_per_segment_matrix', frames_index_per_segment_matrix)
        # frames_index_per_segment_matrix = self.print_me('frames_index_per_segment_matrix', frames_index_per_segment_matrix)


        # remove the columns corresponding to the order from the index matrix
        # this insures that n frames will remain when processing the frame at
        # position [q - (n+1)], where 1 is the window's middle frame
        frames_index_per_segment_trimmed_matrix = frames_index_per_segment_matrix[:, : -self.order * 2]
        # frames_index_per_segment_trimmed_matrix = self.print_me('frames_index_per_segment_trimmed_matrix',frames_index_per_segment_trimmed_matrix)


        # reshaping the index matrix after trimming back to a vector
        frames_index_per_minibatch_trimmed_flattened = frames_index_per_segment_trimmed_matrix.flatten()

        # repeating the flat index a number of times equal to the 2 x order + 1
        frames_index_per_minibatch_trim_flat_tile = T.tile(frames_index_per_minibatch_trimmed_flattened,
                                                           (self.order * 2 + 1, 1))
        # frames_index_per_minibatch_trim_flat_tile = self.print_dim('frames_index_per_minibatch_trim_flat_tile', frames_index_per_minibatch_trim_flat_tile)
        # frames_index_per_minibatch_trim_flat_tile = self.print_me('frames_index_per_minibatch_trim_flat_tile', frames_index_per_minibatch_trim_flat_tile)


        trimmed_vector_length = frames_index_per_minibatch_trimmed_flattened.shape[0]

        order_increments = T.arange((self.order * 2 + 1))  # the increments to

        # order_increments = self.print_me('order_increments', order_increments)

        # repeat each element in the order increments vector a number of times equal to the minibatch trimmed vector
        order_increments_tile = T.tile(order_increments, (trimmed_vector_length, 1))
        # order_increments_tile = self.print_dim('order_increments_tile', order_increments_tile)
        order_increments_tile = order_increments_tile.transpose()
        # the window_index will have a 2n+1 rows, each row has indices of the frames of the segments
        window_index = order_increments_tile + frames_index_per_minibatch_trim_flat_tile
        # window_index = self.print_dim('window_index', window_index)
        # window_index = self.print_me('window_index',window_index)

        result, updates = theano.scan(
            fn=lambda w, i, previous_result, x, mask: previous_result + T.dot(x[i, :], w * mask),
            outputs_info=T.zeros([window_index.shape[1], self.W.shape[2]]),
            sequences=[self.W, window_index],
            non_sequences=[concatenated_segments, self.weightmask],
            n_steps=(self.order * 2 + 1))

        # result = self.print_dim('result', result)
        result = result[-1]

        result = result + self.b
        # result = self.print_dim('result', result)
        activation_input = result.reshape((segment_count, segment_length - self.order * 2, result.shape[1]))  #
        # activation_input = self.print_dim('activation_input', activation_input)
        return self.activation(activation_input)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], input_shape[1] - self.order * 2, self.output_dim)
        # return (1, 1,1)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MaskedConditional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalPooling1D(Layer):
    ''' A single dimensional temporal pooling.
    # Input shape
        3D tensor with shape:  (nb_samples, input_row,input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim=input_dim)`.
    '''

    def __init__(self, output_dim=None, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GlobalPooling1D, self).__init__(**kwargs)

    def print_me(self, name, mat):
        mat = printing.Print('vector')(mat)
        return mat

    def print_dim(self, name, mat):
        mat = printing.Print(name, attrs=['shape'])(mat)
        return mat

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        self.output_dim = input_dim
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_shape[1], input_dim))]

    def call(self, x, mask=None):
        # x = self.print_dim('x', x)
        extra_frames_mean = x.mean(axis=1)
        # result_max = x.max(axis=1)
        # extra_frames_mean = self.print_dim('extra_frames_mean', extra_frames_mean)
        statistics_vector = T.concatenate([extra_frames_mean], axis=1)

        # Add extra dimension
        result = statistics_vector.reshape((statistics_vector.shape[0], 1, statistics_vector.shape[1]))

        return result

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (None, 1, self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(GlobalPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
