import tensorflow as tf

class MetaLearnerNet(tf.contrib.rnn.LayerRNNCell):

    def __init__(self, optimizer, optimizee,
                   activation=None,
                   reuse=None,
                   name=None,
                   dtype=None,
                   **kwargs):
        self.optimizer = optimizer
        self.optimizee = optimizee
        if optimizer.output_size != optimizee.state_size:
            raise ValueError("Unsupported output size of optimizer!")
        super(MetaLearnerNet, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

    @property
    def state_size(self):
        return 2*self.optimizee.output_size + self.optimizer.state_size

    @property
    def output_size(self):
        return 1

    def __build__(self,input_shape):
        self.optimizer.build([input_shape[0],2])
        self.optimizee.build([input_shape[0],input_shape[1]-1])

    def __call__(self, inputs, state):
        full_state = list()
        _, state = self.optimizee.__call__(inputs[:,:-1], state[:,:self.opimizee.output_size])
        kernel = state[:,-self.optimizee.output_size:]
        optimizee_out = tf.matmul(state,kernel)
        optimizer_input = array_ops.concat([optimizee_out,inputs[:,-1:]],axis=1)
        full_state.append(state)
        output, state = self.optimizer.__call__(optimizer_input, state[:,self.optimizee.output_size:-self.optimizee.output_size])
        kernel += output
        full_state.append(kernel)
        full_state.append(state)
        return tf.square(optimizee_out-inputs[:,-1:]), array_ops.concat(full_state,axis=1)