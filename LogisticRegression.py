
from __future__ import print_function

import tensorflow as tf


# Paramters
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("traing_epochs", 25, "Training epochs.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size.")
tf.app.flags.DEFINE_integer("dispaly_step", 1, "Display step.")

FLAGS = tf.app.flags.FLAGS

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Projects\Python\\TensorflowExamples\\', one_hot=True)

# Tf graph input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28 * 28 = 784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition

# Set model weight
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct Model
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

# Init the variables
init = tf.global_variables_initializer()

# Start trainine
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(FLAGS.traing_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/FLAGS.batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x : batch_x, y : batch_y})
            avg_cost += c / total_batch
        if (epoch + 1) % FLAGS.dispaly_step == 0:
            print("Epoch: ", "%04d" % (epoch + 1), ". Cost: ", "{:.9f}".format(avg_cost))
    print("Optimizer Finished.")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))