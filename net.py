import numpy as np
import tensorflow.compat.v1 as tf

import config

tf.disable_eager_execution()

class NN:
    def __init__(self, session):
        self.session = session

    def f_batch(self, state_batch):
        return self.session.run([p_op, v_op], feed_dict={state: state_batch, training: False})

    def train(self, state_batch, pi_batch, z_batch):
        p_loss, v_loss, _ = self.session.run([p_loss_op, v_loss_op, train_step_op], feed_dict={
            state: state_batch,
            pi: pi_batch,
            z: z_batch,
            learning_rate: config.learning_rate,
            training: True})
        return p_loss, v_loss


def init_model():
    x = single_convolutional_block(state)
    for i in range(config.residual_blocks_num):
        x = residual_block(x)
    p_op = policy_head(x)
    v_op = value_head(x)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        v_loss_op, p_loss_op, combined_loss_op = loss(pi, z, p_op, v_op)
        train_step_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config.momentum).minimize(combined_loss_op)
    return p_op, v_op, p_loss_op, v_loss_op, train_step_op


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, kernel_size, filter_num):
    W = weight_variable([kernel_size, kernel_size, x.shape.dims[3].value, filter_num])
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def batch_normalization(x):
    return tf.layers.batch_normalization(x, training=training)


def rectifier_nonlinearity(x):
    return tf.nn.relu(x)


def linear_layer(x, size):
    W = weight_variable([x.shape.dims[1].value, size])
    b = bias_variable([size])
    return tf.matmul(x, W) + b


def single_convolutional_block(x):
    x = conv2d(x, 3, 64)
    x = batch_normalization(x)
    return rectifier_nonlinearity(x)


def residual_block(x):
    original_x = x
    x = conv2d(x, 3, 64)
    x = batch_normalization(x)
    x = rectifier_nonlinearity(x)
    x = conv2d(x, 3, 64)
    x = batch_normalization(x)
    x += original_x
    return rectifier_nonlinearity(x)


def policy_head(x):
    x = conv2d(x, 1, 2)
    x = batch_normalization(x)
    x = rectifier_nonlinearity(x)
    x = tf.reshape(x, [-1, config.board_length * 2])
    return linear_layer(x, config.all_moves_num)


def value_head(x):
    x = conv2d(x, 1, 1)
    x = batch_normalization(x)
    x = rectifier_nonlinearity(x)
    x = tf.reshape(x, [-1, config.board_length])
    x = linear_layer(x, 64)
    x = rectifier_nonlinearity(x)
    x = linear_layer(x, 1)
    return tf.nn.tanh(x)


def loss(pi, z, p, v):
    v_loss = tf.reduce_mean(tf.square(z - v))
    p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=p, labels=pi))
    variables = [v for v in tf.trainable_variables() if 'bias' not in v.name and 'beta' not in v.name]
    l2 = tf.add_n([tf.nn.l2_loss(variable) for variable in variables])
    return v_loss, p_loss, v_loss + p_loss + config.l2_weight * l2


state = tf.placeholder(tf.float32, [None, config.N, config.N, config.history_num * 2 + 1], name='state')
pi = tf.placeholder(tf.float32, [None, config.all_moves_num], name='pi')
z = tf.placeholder(tf.float32, [None, 1], name='z')
training = tf.placeholder(tf.bool, name='training')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
p_op, v_op, p_loss_op, v_loss_op, train_step_op = init_model()
