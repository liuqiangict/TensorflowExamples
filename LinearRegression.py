
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rand = np.random

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("training_epochs", 1000, "Training epochs.")
tf.app.flags.DEFINE_integer("desplay_step", 50, "Display steps.")

FLAGS = tf.app.flags.FLAGS


# Training data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# Tf graph input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weight
W = tf.Variable(rand.randn(), name = "weight")
b = tf.Variable(rand.randn(), name = "bias")


# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Loss, mean squared error
loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Start Training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for epoch in range(FLAGS.training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict = {X : x, Y : y})
        if (epoch + 1) % FLAGS.desplay_step == 0:
            l = sess.run(loss, feed_dict = {X : train_X, Y : train_Y})
            print("Epoch: ", "%04d" % (epoch + 1), ", Loss: " "{:.9f}".format(l))

    print("Optimizer finished.")
    training_loss = sess.run(loss, feed_dict = {X : train_X, Y : train_Y})
    print("Training loss: ", training_loss, ", W: ", sess.run(W), ", b: ", sess.run(b))

    # Graphic display
    plt.plot(train_X, train_Y, "ro", label = "Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label = "Fitted line")
    plt.legend()
    plt.show()

    # Testing
    print("Testing.")
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    testing_loss = sess.run(tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * test_X.shape[0]), feed_dict={X: test_X, Y: test_Y})
    print("Testing loss: ", testing_loss)
    print("Absolute mean square loss difference:", abs(training_loss - testing_loss))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()