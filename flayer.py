from keras.engine.topology import Layer
import keras.backend as K

class MSETLayer(Layer):
    """
    """
    def __init__(self, kernel_size, **kwargs):
        self.kernel_size=kernel_size
        super(MSETLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MSETLayer, self).build(input_shape)

    def call(self, x):
        inp_shape = K.tf.shape(x)
        def get_output(j):
            def get_time(i):
                def get_new_x(w):
                    def get_columns(h):
                        pixel = K.tf.squared_difference(K.tf.slice(x,
                                                                    begin=[j,i,w,h,0],
                                                                    size=[1,1,self.kernel_size, self.kernel_size,inp_shape[4]])
                                                        ,K.tf.slice(x,
                                                                    begin=[j,i+1,w,h,0],
                                                                    size=[1,1,self.kernel_size, self.kernel_size,inp_shape[4]])
                                )
                        return K.tf.reshape(K.tf.reduce_mean(pixel), shape=[1])
                    return K.tf.map_fn(get_columns, K.tf.range(start=0,
                                                               limit=inp_shape[3]//self.kernel_size),
                                        swap_memory=True,
                                        back_prop=False,
                                        parallel_iterations=10,
                                        infer_shape=True,
                                        dtype=K.tf.float32
                            )
                return K.tf.map_fn(get_new_x, K.tf.range(start=0,
                                                         limit=inp_shape[2]//self.kernel_size),
                                    swap_memory=True,
                                    back_prop=False,
                                    parallel_iterations=10,
                                    infer_shape=True,
                                    dtype=K.tf.float32
                        )
            return K.tf.map_fn(get_time, K.tf.range(start=0,
                                                    limit=inp_shape[1]-1),
                                swap_memory=True,
                                back_prop=False,
                                parallel_iterations=2,
                                infer_shape=True,
                                dtype=K.tf.float32
                    )
        out = K.tf.map_fn(get_output, K.tf.range(start=0,
                                                 limit=inp_shape[0]),
                          swap_memory=True,
                          back_prop=False,
                          parallel_iterations=2,
                          infer_shape=True,
                          dtype=K.tf.float32)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1]-1,
                input_shape[2]//self.kernel_size,
                input_shape[3]//self.kernel_size,
                input_shape[4])
