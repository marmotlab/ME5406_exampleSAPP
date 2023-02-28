import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

# parameters for training
GRAD_CLIP = 1000.0
RNN_SIZE = 128
GOAL_REPR_SIZE = 12


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class ACNet(keras.Model):
    def __init__(self, a_size, trainer, TRAINING, NUM_CHANNEL, GRID_SIZE):
        super(ACNet, self).__init__()
        self.loss = 0.0
        self.value_loss = 0.0
        self.entropy = 0.0
        self.policy_loss = 0.0
        self.loss = 0.0
        self.trainer = trainer
        self.policy = []
        self.num_channel = NUM_CHANNEL
        self.grid_size = GRID_SIZE
        self._build_net(RNN_SIZE, a_size)

    @tf.function
    def call(self, inputs, goal_pos):
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        p1 = self.pool1(self.conv1b(self.conv1a(self.conv1(inputs))))
        p2 = self.pool2(self.conv2b(self.conv2a(self.conv2(p1))))
        conv3 = self.conv3(p2)
        flat = tf.nn.relu(self.flat(conv3))
        gl = tf.nn.relu(self.goal_layer(goal_pos))
        hidden_input = tf.concat([flat, gl], 1)
        d2 = self.h2(self.h1(hidden_input))

        h3 = tf.nn.relu(d2 + hidden_input)
        rnn_in = tf.expand_dims(h3, [0])
        lstm_outputs = self.LSTM(rnn_in)
        rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])
        pl = self.policy_layer(rnn_out)
        policy = tf.nn.softmax(pl)
        value = self.value_layer(rnn_out)

        return policy, value

    def _build_net(self, RNN_SIZE, a_size):
        self.a_size = a_size
        self.conv1 = layers.Conv2D(padding="same", filters=RNN_SIZE // 4, kernel_size=[3, 3], strides=1, data_format="channels_last", kernel_initializer="he_normal", activation=tf.nn.relu)
        self.conv1a = layers.Conv2D(padding="same", filters=RNN_SIZE // 4, kernel_size=[3, 3], strides=1, data_format="channels_last", kernel_initializer="he_normal", activation=tf.nn.relu)
        self.conv1b = layers.Conv2D(padding="same", filters=RNN_SIZE // 4, kernel_size=[3, 3], strides=1, data_format="channels_last", kernel_initializer="he_normal", activation=tf.nn.relu)
        self.pool1 = layers.MaxPool2D(pool_size=[2, 2])
        self.conv2 = layers.Conv2D(padding="same", filters=RNN_SIZE // 2, kernel_size=[3, 3], strides=1, data_format="channels_last", kernel_initializer="he_normal", activation=tf.nn.relu)
        self.conv2a = layers.Conv2D(padding="same", filters=RNN_SIZE // 2, kernel_size=[3, 3], strides=1, data_format="channels_last", kernel_initializer="he_normal", activation=tf.nn.relu)
        self.conv2b = layers.Conv2D(padding="same", filters=RNN_SIZE // 2, kernel_size=[3, 3], strides=1, data_format="channels_last", kernel_initializer="he_normal", activation=tf.nn.relu)
        self.pool2 = layers.MaxPool2D(pool_size=[2, 2])
        self.conv3 = layers.Conv2D(padding="valid", filters=RNN_SIZE - GOAL_REPR_SIZE, kernel_size=[2, 2], strides=1, data_format="channels_last", kernel_initializer="he_normal", activation=None)
        self.flat = layers.Flatten()
        self.goal_layer = layers.Dense(units=GOAL_REPR_SIZE)

        self.h1 = layers.Dense(units=RNN_SIZE, activation='relu')
        self.h2 = layers.Dense(units=RNN_SIZE, activation=None)
        self.LSTM = layers.LSTM(units=RNN_SIZE, return_sequences=True, return_state=False, stateful=True)

        self.policy_layer = layers.Dense(units=a_size, kernel_initializer="random_normal", activation=None)
        self.value_layer = layers.Dense(units=1, kernel_initializer="random_normal", activation=None)

    def compute_loss(self, actions, train_valid, target_v, advantages, inputs, goal_pos):
        inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        p1 = self.pool1(self.conv1b(self.conv1a(self.conv1(inputs))))
        p2 = self.pool2(self.conv2b(self.conv2a(self.conv2(p1))))
        conv3 = self.conv3(p2)
        flat = tf.nn.relu(self.flat(conv3))
        gl = tf.nn.relu(self.goal_layer(goal_pos))
        hidden_input = tf.concat([flat, gl], 1)
        d2 = self.h2(self.h1(hidden_input))

        h3 = tf.nn.relu(d2 + hidden_input)
        rnn_in = tf.expand_dims(h3, [0])
        lstm_outputs = self.LSTM(rnn_in)
        rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])
        pl = self.policy_layer(rnn_out)
        policy = tf.nn.softmax(pl)
        valids = tf.sigmoid(pl)
        value = self.value_layer(rnn_out)
        actions_onehot = tf.one_hot(actions, self.a_size, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])

        # Loss Functions
        value_loss = 0.5 * tf.reduce_sum(tf.square(target_v - tf.reshape(value, shape=[-1])))
        entropy = - 0.01 * tf.reduce_sum(policy * tf.math.log(tf.clip_by_value(policy, 1e-10, 1.0)))
        policy_loss = - tf.reduce_sum(tf.math.log(tf.clip_by_value(responsible_outputs, 1e-15, 1.0)) * advantages)
        valid_loss = - 0.5 * tf.reduce_sum(tf.math.log(tf.clip_by_value(valids, 1e-10, 1.0)) * train_valid + tf.math.log(tf.clip_by_value(1 - valids, 1e-10, 1.0)) * (1 - train_valid))
        var_norms = tf.linalg.global_norm(self.trainable_weights)
        loss = value_loss + policy_loss - entropy + valid_loss

        return loss, value_loss, policy_loss, entropy, var_norms, valid_loss

    def apply_gradients(self, gradients):
        clippedGrads, norms = tf.clip_by_global_norm(gradients, GRAD_CLIP)
        self.trainer.apply_gradients(zip(clippedGrads, self.trainable_weights))

    def initial_feed(self):
        obs = np.random.random((1, self.num_channel, self.grid_size, self.grid_size))
        scalars = np.random.random((1, 3))
        self.call(tf.convert_to_tensor(obs, dtype=tf.float32), tf.convert_to_tensor(scalars, dtype=tf.float32))
