from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import _generate_dropout_mask
import numpy as np
import warnings
from keras.engine.topology import InputSpec, Layer
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.layers import Recurrent, ConvRecurrent2D
from keras.layers.recurrent import RNN
from keras.utils.generic_utils import has_arg

# keras.io
def _standardize_args(inputs, initial_state, constants, num_constants):
    if isinstance(inputs, list):
        assert initial_state is None and constants is None
        if num_constants is not None:
            constants = inputs[-num_constants:]
            inputs = inputs[:-num_constants]
        if len(inputs) > 1:
            initial_state = inputs[1:]
        inputs = inputs[0]

    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    initial_state = to_list_or_none(initial_state)
    constants = to_list_or_none(constants)

    return inputs, initial_state, constants

class DSConvRNN2D(RNN):
    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if unroll:
            raise TypeError('Unrolling isn\'t possible with '
                            'convolutional RNNs.')
        if isinstance(cell, (list, tuple)):
            # The StackedConvRNN2DCells isn't implemented yet.
            raise TypeError('It is not possible at the moment to'
                            'stack convolutional cells.')
        super(DSConvRNN2D, self).__init__(cell,
                                        return_sequences,
                                        return_state,
                                        go_backwards,
                                        stateful,
                                        unroll,
                                        **kwargs)
        self.input_spec = [InputSpec(ndim=5)]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        cell = self.cell
        if cell.data_format == 'channels_first':
            rows = input_shape[3]
            cols = input_shape[4]
        elif cell.data_format == 'channels_last':
            rows = input_shape[2]
            cols = input_shape[3]
        rows = conv_utils.conv_output_length(rows,
                                             cell.kernel_size[0],
                                             padding=cell.padding,
                                             stride=cell.strides[0],
                                             dilation=cell.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols,
                                             cell.kernel_size[1],
                                             padding=cell.padding,
                                             stride=cell.strides[1],
                                             dilation=cell.dilation_rate[1])

        if cell.data_format == 'channels_first':
            output_shape = input_shape[:2] + (cell.filters, rows, cols)
        elif cell.data_format == 'channels_last':
            output_shape = input_shape[:2] + (rows, cols, cell.filters)

        if not self.return_sequences:
            output_shape = output_shape[:1] + output_shape[2:]

        if self.return_state:
            output_shape = [output_shape]
            if cell.data_format == 'channels_first':
                output_shape += [(input_shape[0], cell.filters, rows, cols)
                                 for _ in range(2)]
            elif cell.data_format == 'channels_last':
                output_shape += [(input_shape[0], rows, cols, cell.filters)
                                 for _ in range(2)]
        return output_shape

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if self.cell.data_format == 'channels_first':
                ch_dim = 1
            elif self.cell.data_format == 'channels_last':
                ch_dim = 3
            if not [spec.shape[ch_dim] for spec in self.state_spec] == state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format([spec.shape for spec in self.state_spec], self.cell.state_size))
        else:
            if self.cell.data_format == 'channels_first':
                self.state_spec = [InputSpec(shape=(None, dim, None, None))
                                   for dim in state_size]
            elif self.cell.data_format == 'channels_last':
                self.state_spec = [InputSpec(shape=(None, None, None, dim))
                                   for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # (samples, timesteps, rows, cols, filters)
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, filters)
        initial_state = K.sum(initial_state, axis=1)
        shape = list(self.cell.kernel_shape)
        shape_1x1 = list(self.cell.kernel_1x1_shape)
        shape[-1] = self.cell.depth_multiplier
        shape_1x1[-1] = self.cell.filters
        initial_state = self.cell.input_conv(initial_state,
                                             K.zeros(tuple(shape)),
                                             K.zeros(tuple(shape_1x1)),
                                             padding=self.cell.padding)
        # Fix for Theano because it needs
        # K.int_shape to work in call() with initial_state.
        keras_shape = list(K.int_shape(inputs))
        keras_shape.pop(1)
        if K.image_data_format() == 'channels_first':
            indices = 2, 3
        else:
            indices = 1, 2
        for i, j in enumerate(indices):
            keras_shape[j] = conv_utils.conv_output_length(
                keras_shape[j],
                shape[i],
                padding=self.cell.padding,
                stride=self.cell.strides[i],
                dilation=self.cell.dilation_rate[i])
        initial_state._keras_shape = keras_shape

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(DSConvRNN2D, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = []
            for state in initial_state:
                try:
                    shape = K.int_shape(state)
                # Fix for Theano
                except TypeError:
                    shape = tuple(None for _ in range(K.ndim(state)))
                self.state_spec.append(InputSpec(shape=shape))

            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != K.is_keras_tensor(additional_inputs[0]):
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors')

        if K.is_keras_tensor(additional_inputs[0]):
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(DSConvRNN2D, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(DSConvRNN2D, self).__call__(inputs, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        timesteps = K.int_shape(inputs)[1]

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        input_shape = self.input_spec[0].shape
        state_shape = self.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = state_shape[0]
        if self.return_sequences:
            state_shape = state_shape[:1] + state_shape[2:]
        if None in state_shape:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.\n'
                             'The same thing goes for the number of rows and columns.')

        # helper function
        def get_tuple_shape(nb_channels):
            result = list(state_shape)
            if self.cell.data_format == 'channels_first':
                result[1] = nb_channels
            elif self.cell.data_format == 'channels_last':
                result[3] = nb_channels
            else:
                raise KeyError
            return tuple(result)

        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros(get_tuple_shape(dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros(get_tuple_shape(dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros(get_tuple_shape(self.cell.state_size)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                                         'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != get_tuple_shape(dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str(get_tuple_shape(dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
        K.set_value(state, value)

class DepthwiseSepConvLSTM2DCell(Layer):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 use_1x1_bias=True,
                 depth_multiplier=1,
                 kernel_initializer='glorot_uniform',
                 kernel_1x1_initializer = 'glorot_uniform',
                 recurrent_initializer='orthogonal',
                 recurrent_1x1_initializer='orthoganal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 kernel_1x1_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 kernel_1x1_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):

        super(DepthwiseSepConvLSTM2DCell, self).__init__(**kwargs)
        self.kernel_1x1_regularizer = regularizers.get(kernel_1x1_regularizer)
        self.kernel_1x1_initializer = initializers.get(kernel_1x1_initializer)
        self.kernel_1x1_constraint = constraints.get(kernel_1x1_constraint)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.depth_multiplier = depth_multiplier

        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier * 4)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size +\
                                            (self.filters, self.depth_multiplier * 4)
        kernel_1x1_shape = (1, 1)+(input_dim, self.filters*4)
        recurrent_kernel_1x1_shape = (1, 1)+(self.filters, self.filters*4)
        self.kernel_1x1_shape = kernel_1x1_shape
        self.recurrent_kernel_1x1 = recurrent_kernel_1x1_shape


        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.kernel_1x1 = self.add_weight(shape=kernel_1x1_shape,
                                      initializer=self.kernel_1x1_initializer,
                                      name='depthwise_kernel',
                                      regularizer=self.kernel_1x1_regularizer,
                                      constraint=self.kernel_1x1_constraint)
        self.recurrent_kernel_1x1 = self.add_weight(
            shape=recurrent_kernel_1x1_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_1x1_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters,), *args, **kwargs),
                        initializers.Ones()((self.filters,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.filters * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[
                                    :,
                                    :,
                                    :,
                                    :self.depth_multiplier]
        self.recurrent_kernel_i = self.recurrent_kernel[
                                    :,
                                    :,
                                    :,
                                    :self.depth_multiplier]
        self.kernel_f = self.kernel[
                                    :,
                                    :,
                                    :,
                                    self.depth_multiplier: self.depth_multiplier * 2]
        self.recurrent_kernel_f = self.recurrent_kernel[
                                    :,
                                    :,
                                    :,
                                    self.depth_multiplier: self.depth_multiplier * 2]
        self.kernel_c = self.kernel[
                                    :,
                                    :,
                                    :,
                                    self.depth_multiplier * 2: self.depth_multiplier * 3]
        self.recurrent_kernel_c = self.recurrent_kernel[
                                    :,
                                    :,
                                    :,
                                    self.depth_multiplier * 2: self.depth_multiplier * 3]
        self.kernel_o = self.kernel[
                                    :,
                                    :,
                                    :,
                                    self.depth_multiplier * 3:]
        self.recurrent_kernel_o = self.recurrent_kernel[
                                    :,
                                    :,
                                    :,
                                    self.depth_multiplier * 3:]
        self.kernel_1x1_i = self.kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    :self.filters]
        self.recurrent_kernel_1x1_i = self.recurrent_kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    :self.filters]
        self.kernel_1x1_f = self.kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    self.filters: self.filters * 2]
        self.recurrent_kernel_1x1_f = self.recurrent_kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    self.filters:self.filters * 2]
        self.kernel_1x1_c = self.kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    self.filters * 2: self.filters * 3]
        self.recurrent_kernel_1x1_c = self.recurrent_kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    self.filters * 2: self.filters * 3]
        self.kernel_1x1_o = self.kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    self.filters * 3:]
        self.recurrent_kernel_1x1_o = self.recurrent_kernel_1x1[
                                    :,
                                    :,
                                    :,
                                    self.filters * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.filters]
            self.bias_f = self.bias[self.filters: self.filters * 2]
            self.bias_c = self.bias[self.filters * 2: self.filters * 3]
            self.bias_o = self.bias[self.filters * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[1]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        x_i = self.input_conv(inputs_i, w=self.kernel_i,
                              w_1x1=self.kernel_1x1_i, b=self.bias_i,
                              padding=self.padding)
        x_f = self.input_conv(inputs_f, w=self.kernel_f,
                              w_1x1=self.kernel_1x1_f, b=self.bias_f,
                              padding=self.padding)
        x_c = self.input_conv(inputs_c, w=self.kernel_c,
                              w_1x1=self.kernel_1x1_c, b=self.bias_c,
                              padding=self.padding)
        x_o = self.input_conv(inputs_o, w=self.kernel_o,
                              w_1x1=self.kernel_1x1_o, b=self.bias_o,
                              padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i,
                                  self.recurrent_kernel_i,
                                  self.recurrent_kernel_1x1_i)
        h_f = self.recurrent_conv(h_tm1_f,
                                  self.recurrent_kernel_f,
                                  self.recurrent_kernel_1x1_f)
        h_c = self.recurrent_conv(h_tm1_c,
                                  self.recurrent_kernel_c,
                                  self.recurrent_kernel_1x1_c)
        h_o = self.recurrent_conv(h_tm1_o,
                                  self.recurrent_kernel_o,
                                  self.recurrent_kernel_1x1_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]

    def input_conv(self, x, w, w_1x1, b=None, padding='valid'):
        if 'last' in self.data_format:
            dformat = 'NHWC'
        else:
            dformat = 'NCHW'
        depthwise = K.tf.nn.depthwise_conv2d(x,
                                          w,
                                          strides=(1, self.strides[0], self.strides[1], 1),
                                          padding='SAME',
                                          rate=self.dilation_rate,
                                          data_format=dformat)
        conv_out = K.conv2d(depthwise,
                            w_1x1,
                            strides=(1, 1),
                            padding=padding,
                            data_format=self.data_format)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w, w_1x1):
        if 'last' in self.data_format:
            dformat = 'NHWC'
        else:
            dformat = 'NCHW'
        depthwise = K.tf.nn.depthwise_conv2d(x,
                                          w,
                                          strides=(1, 1, 1, 1),
                                          padding= 'SAME',
                                          rate=self.dilation_rate,
                                          data_format=dformat)
        conv_out = K.conv2d(depthwise,
                            w_1x1,
                            strides=(1, 1),
                            padding= 'same',
                            data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'kernel_1x1_initializer':initializers.serialize(self.kernel_1x1_initializer),
                  'kernel_1x1_regularizer':regularizers.serialize(self.kernel_1x1_regularizer),
                  'kernel_1x1_constraint':constraints.serialize(self.kernel_1x1_constraint),
                  'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout
                  }
        base_config = super(DepthwiseSepConvLSTM2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DepthwiseSepConvLSTM2D(DSConvRNN2D):
    @interfaces.legacy_convlstm2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_1x1_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 depth_multiplier=1,
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 kernel_1x1_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 kernel_1x1_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = DepthwiseSepConvLSTM2DCell(
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              data_format=data_format,
                              dilation_rate=dilation_rate,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              use_bias=use_bias,
                              depth_multiplier=1,
                              kernel_initializer=kernel_initializer,
                              kernel_1x1_initializer=kernel_1x1_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              unit_forget_bias=unit_forget_bias,
                              kernel_regularizer=kernel_regularizer,
                              kernel_1x1_regularizer=kernel_1x1_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              kernel_1x1_constraint=kernel_1x1_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout)
        super(DepthwiseSepConvLSTM2D, self).__init__(cell,
                                         return_sequences=return_sequences,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(DepthwiseSepConvLSTM2D, self).call(
                                                inputs,
                                                mask=mask,
                                                training=training,
                                                initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def depth_multiplier(self):
        return self.cell.depth_multiplier

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def kernel_1x1_initializer(self):
        return self.cell.kernel_1x1_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def kernel_1x1_regularizer(self):
        return self.cell.kernel_1x1_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def kernel_1x1_constraint(self):
        return self.cell.kernel_1x1_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_1x1_initializer': initializers.serialize(self.kernel_1x1_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'kernel_1x1_regularizer': regularizers.serialize(self.kernel_1x1_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'kernel_1x1_constraint': constraints.serialize(self.kernel_1x1_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(DepthwiseSepConvLSTM2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
