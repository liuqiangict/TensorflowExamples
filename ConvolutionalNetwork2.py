
from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Projects\Python\\TensorflowExamples\\', one_hot=False)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("num_steps", 2000, "Number steps.")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
FLAGS = tf.app.flags.FLAGS

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        maxpool1 = tf.layers.max_pooling2d(conv1, 2, 2,)

        conv2 = tf.layers.conv2d(maxpool1, 64, 3, activation=tf.nn.relu)
        maxpool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(maxpool2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function
def model_fn(features, labels, mode):
    logits_train = conv_net(features, num_classes, dropout, reuse = False, is_training = True)
    logits_test = conv_net(features, num_classes, dropout, reuse = True, is_training = False)

    pred_classes = tf.arg_max(logits_test, 1)
    pred_prob = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels = tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels = labels, predictions = pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy' : acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images}, y=mnist.train.labels, batch_size=FLAGS.batch_size, num_epochs=None, shuffle=True)

model.train(input_fn, steps = FLAGS.num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=FLAGS.batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)
print("Testing Accuracy:", e['accuracy'])