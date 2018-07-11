
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("num_steps", 10000, "Number steps.")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("display_step", 200, "Display steps.")
FLAGS = tf.app.flags.FLAGS

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Projects\Python\\TensorflowExamples\\', one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
# Network Paramters
num_input = 28
time_step = 28
num_hidden = 128
num_classes = 10

# tf graph 
X = tf.placeholder("float", [None, time_step, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out' : tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
biases = {
    'out' : tf.Variable(tf.random_normal([num_classes]))
}

# Contruct model
def RNN(x, weights, biases):
    x = tf.unstack(x, time_step, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
pred = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate the model
correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializer the variables
init = tf.global_variables_initializer();

# Start trainings
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, FLAGS.num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
        batch_x = batch_x.reshape((FLAGS.batch_size, time_step, num_input))
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})
        if step % FLAGS.display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X:batch_x, Y : batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc)) 
    print("Optimizer finished.")
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, time_step, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))