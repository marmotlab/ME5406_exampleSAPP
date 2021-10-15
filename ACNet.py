import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


#parameters for training
GRAD_CLIP              = 1000.0
RNN_SIZE               = 128
GOAL_REPR_SIZE         = 12


#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class ACNet:
    def __init__(self, scope, a_size, trainer, TRAINING, GRID_SIZE):
        with tf.variable_scope(str(scope)+'/qvalues'):
            #The input size may require more work to fit the interface.
            self.inputs   = tf.placeholder(shape=[None,3,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.goal_pos = tf.placeholder(shape=[None,3],dtype=tf.float32)
            self.myinput  = tf.transpose(self.inputs, perm=[0,2,3,1])
            
            # Define initial and current LSTM states
            c_init = np.zeros((1, RNN_SIZE), np.float32) # 1x128 full of zeros
            h_init = np.zeros((1, RNN_SIZE), np.float32) # 1x128 full of zeros
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, RNN_SIZE]) # 1x128, current state/memory
            h_in = tf.placeholder(tf.float32, [1, RNN_SIZE]) # 1x128, current state/memory
            #self.state_in = (c_in, h_in)
            self.state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)

            self.policy, self.value, self.state_out, self.valids = self._build_net(self.myinput,self.goal_pos,RNN_SIZE,a_size)

        if TRAINING:
            self.actions                = tf.placeholder(shape=[None], dtype=tf.int32) # [a_1, a_2, ..., a_T], ex [2,3]
            self.actions_onehot         = tf.one_hot(self.actions, a_size, dtype=tf.float32) # [ [0,0,1,0,0], [0,0,0,1,0] ]
            self.train_valid            = tf.placeholder(shape=[None,a_size], dtype=tf.float32)  # [1,0,0,1,0]
            self.target_v               = tf.placeholder(tf.float32, [None], 'Vtarget') # [v_1, v_2, v_3, ..., v_T]
            self.advantages             = tf.placeholder(shape=[None], dtype=tf.float32) # [A_1, A_2, ..., A_T]
            self.responsible_outputs    = tf.reduce_sum(self.policy * self.actions_onehot, [1]) # [ p(a_1), p(a_2), .., p(a_T) ]

            # Loss Functions
            self.value_loss    = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
            self.entropy       = - 0.01 * tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-10,1.0)))
            self.policy_loss   = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.advantages)
            self.valid_loss    = - 0.5 * tf.reduce_sum(tf.log(tf.clip_by_value(self.valids,1e-10,1.0)) *\
                                self.train_valid + tf.log(tf.clip_by_value(1 - self.valids,1e-10,1.0)) * (1 - self.train_valid))
            self.loss          = self.value_loss + self.policy_loss - self.entropy + self.valid_loss

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            trainable_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/qvalues')
            self.gradients     = tf.gradients(self.loss, trainable_vars)
            self.var_norms     = tf.global_norm(trainable_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)
            self.apply_grads   = trainer.apply_gradients(zip(grads, trainable_vars))
        print("Hello World... From  "+str(scope))     # :)

    def _build_net(self,inputs,goal_pos,RNN_SIZE,a_size):
        w_init   = layers.variance_scaling_initializer()
        
        # input: 11x11x3
        # goal_pos = 1x3
        
        conv1    =  layers.conv2d(inputs=inputs,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init, activation_fn=tf.nn.relu) # 11x11x32
        conv1a   =  layers.conv2d(inputs=conv1,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init, activation_fn=tf.nn.relu) # 11x11x32
        conv1b   =  layers.conv2d(inputs=conv1a,   padding="SAME", num_outputs=RNN_SIZE//4,  kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init, activation_fn=tf.nn.relu) # 11x11x32
        pool1    =  layers.max_pool2d(inputs=conv1b, kernel_size=[2,2]) # 5x5x32

        conv2    =  layers.conv2d(inputs=pool1,    padding="SAME", num_outputs=RNN_SIZE//2, kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu) # 5x5x64
        conv2a   =  layers.conv2d(inputs=conv2,    padding="SAME", num_outputs=RNN_SIZE//2, kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu) # 5x5x64
        conv2b   =  layers.conv2d(inputs=conv2a,    padding="SAME", num_outputs=RNN_SIZE//2, kernel_size=[3, 3],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu) # 5x5x64
        pool2    =  layers.max_pool2d(inputs=conv2b, kernel_size=[2,2])   # 2x2x64
        conv3    =  layers.conv2d(inputs=pool2,    padding="VALID", num_outputs=RNN_SIZE-GOAL_REPR_SIZE, kernel_size=[2, 2],   stride=1, data_format="NHWC", weights_initializer=w_init,activation_fn=None)       # 1x1x116

        flat         = tf.nn.relu(layers.flatten(conv3)) # 1x116
        goal_layer   = layers.fully_connected(inputs=goal_pos, num_outputs=GOAL_REPR_SIZE) # 1x12 (WITH ReLU)
        hidden_input = tf.concat([flat, goal_layer],1) # # 1x116 + # 1x12 = # 1x128
        h1 = layers.fully_connected(inputs=hidden_input,  num_outputs=RNN_SIZE) # 1x128 (WITH ReLU)
        h2 = layers.fully_connected(inputs=h1,  num_outputs=RNN_SIZE, activation_fn=None) # 1x128 (WITHOUT ReLU)
        self.h3 = tf.nn.relu(h2 + hidden_input) # residual shortcut, # 1x128

        #Recurrent network for temporal dependencies
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE,state_is_tuple=True)

        rnn_in = tf.expand_dims(self.h3, [0])
        step_size = tf.shape(inputs)[:1]
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=self.state_in, sequence_length=step_size,
            time_major=False) # THE lstm operation (rnn_in (h3) "+" current LSTM state (state_in) --> lstm_outputs, new state (lstm_state)
        lstm_c, lstm_h = lstm_state
        state_out = (lstm_c[:1, :], lstm_h[:1, :]) # new LSTM state/memory
        self.rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE]) # output of the LSTM (from h3 "+" LSTM state) # 1x128

        policy_layer = layers.fully_connected(inputs=self.rnn_out, num_outputs=a_size, weights_initializer=normalized_columns_initializer(1./float(a_size)), biases_initializer=None, activation_fn=None) # [w1 w2 w3 w4 w5]
        policy       = tf.nn.softmax(policy_layer) # 1x5 of action activation (normalized probability vector) # [p1 p2 p3 p4 p5] sum(pi) = 1
        policy_sig   = tf.sigmoid(policy_layer)                                                              # [b1 b2 b3 b4 b5] bi \in {0,1}
        value        = layers.fully_connected(inputs=self.rnn_out, num_outputs=1, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None) # 1x1

        return policy, value, state_out, policy_sig
