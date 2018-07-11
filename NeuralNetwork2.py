
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("num_steps", 1000, "Number steps.")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("display_step", 10, "Display steps.")
FLAGS = tf.app.flags.FLAGS

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Projects\Python\\TensorflowExamples\\', one_hot=False)

# Network paramters
n_hidden_1 = 320
n_hidden_2 = 160
num_input = 784
num_classes = 10

# Define the neural network
def neural_net(x_dict):
    x = x_dict['images']
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

# Define the model function
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Prediction
    pred_classes = tf.arg_max(logits, dimension = 1)
    pred_probas = tf.nn.softmax(logits)

    # Early return for prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.learning_rate)
    train_op =optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy', acc_op})

    return estim_specs


# Build model
model = tf.estimator.Estimator(model_fn)

# Define input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images' : mnist.train.images}, y=mnist.train.labels, batch_size=FLAGS.batch_size, num_epochs=None, shuffle=True)

# Train model
model.train(input_fn, steps=FLAGS.num_steps)

# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])