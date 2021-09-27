"""Learned primal-dual method."""

import os
import adler
adler.util.gpu.setup_one_gpu()

from phantom import random_phantom_channels
from adler.odl.phantom import random_phantom
from adler.tensorflow import prelu, cosine_decay

import tensorflow as tf
import numpy as np
import odl
from layer import as_tensorflow_layer

np.random.seed(0)
name = os.path.splitext(os.path.basename(__file__))[0]

sess = tf.InteractiveSession()

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator

# Create tensorflow layer from odl operator
odl_op_layer = as_tensorflow_layer(operator,
                                        'RayTransform')
odl_op_layer_adjoint = as_tensorflow_layer(operator.adjoint,
                                        'RayTransformAdjoint')

# User selected paramters
n_data = 5
n_iter = 10
n_primal = 5
n_dual = 5
n_channels = 2

def generate_data(validation=False, n_channels=1, n_data=1):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    y_arr = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], n_channels), dtype='float32')
    x_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], n_channels), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = random_phantom_channels(space, n_channels = n_channels)
        if n_channels == 1:
            data = operator(phantom)
            noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05
            x_true_arr[i, ..., 0] = phantom
            y_arr[i, ..., 0] = noisy_data
        else:
            for j in range(n_channels):
                data = operator(phantom[j])
                noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05
                x_true_arr[i, ..., j] = phantom[j]
                y_arr[i, ..., j] = noisy_data

    return y_arr, x_true_arr


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, n_channels], name="x_true")
    y_rt = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], n_channels], name="y_rt")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([tf.zeros_like(x_true)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(y_rt)] * n_dual, axis=-1)


    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            # need to change this so there a n_channels of evalop
            evalop1 = odl_op_layer(primal[..., 1:2])
            evalop2 = odl_op_layer(primal[..., 2:3])
            update = tf.concat([dual, evalop1, evalop2, y_rt], axis=-1)
            # THE NETWORK BELOW
            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_dual*n_channels)
            print(np.shape(primal))
            dual = dual + update
            print(np.shape(primal))

        with tf.variable_scope('primal_iterate_{}'.format(i)):
            # need to change this so there a n_channels of evalop
            evalop1 = odl_op_layer_adjoint(dual[..., 0:1])
            evalop2 = odl_op_layer_adjoint(dual[..., 1:2])
            update = tf.concat([primal, evalop1, evalop2], axis=-1)
            # THE NETWORK BELOW
            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_primal*n_channels)
            primal = primal + update


    x_result = primal[..., 0:n_channels]


with tf.name_scope('loss'):
    residual = x_result - x_true
    squared_error = residual ** 2
    loss = tf.reduce_mean(squared_error)


with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 100001
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta2=0.99)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


# Summaries
# tensorboard --logdir=...

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('psnr', -10 * tf.log(loss) / tf.log(10.0))

    tf.summary.image('x_result', x_result)
    tf.summary.image('x_true', x_true)
    tf.summary.image('squared_error', squared_error)
    tf.summary.image('residual', residual)

    merged_summary = tf.summary.merge_all()
    test_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/test',
                                                sess.graph)
    train_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/train')

# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
y_arr_validate, x_true_arr_validate = generate_data(n_channels=n_channels, n_data = 1)

if 0:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Train the network
for i in range(0, maximum_steps):
    if i%10 == 0:
        y_arr, x_true_arr = generate_data(n_channels=n_channels, n_data = 5)
    _, merged_summary_result_train, global_step_result = sess.run([optimizer, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr,
                                         y_rt: y_arr,
                                         is_training: True})

    if i>0 and i%10 == 0:
        loss_result, merged_summary_result, global_step_result = sess.run([loss, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr_validate,
                                         y_rt: y_arr_validate,
                                         is_training: False})

        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        test_summary_writer.add_summary(merged_summary_result, global_step_result)

        print('iter={}, loss={}'.format(global_step_result, loss_result))

    if i>0 and i%1000 == 0:
        saver.save(sess,
                   adler.tensorflow.util.default_checkpoint_path(name))
