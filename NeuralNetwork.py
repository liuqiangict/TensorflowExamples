
from __future__ import print_function

import tensorflow as tf

# Paramter
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Leaning rate.")
tf.app.flags.DEFINE_integer("num_steps", 5000, "Number steps.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size.")
tf.app.flags.DEFINE_integer("display_step", 100, "Dispaly step.")
FLAGS = tf.app.flags.FLAGS

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Projects\Python\\TensorflowExamples\\', one_hot=True)

# Network definition
n_hidden_1 = 320 # 1st layer number of neurons
n_hidden_2 = 160 # 2nd layer number of neurons
num_input = 784  # MNIST data input, 28 * 28
num_calsses = 10 # MNIST total classes

# tf graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_calsses])

# Store layers weight & biad
weights = {
    'h1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_hidden_2, num_calsses]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([num_calsses]))
}

# Create model
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
pred = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Start training
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, FLAGS.num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
        sess.run(train_op, feed_dict = {X : batch_x, Y : batch_y})
        if step % FLAGS.display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict = {X : batch_x, Y : batch_y})
            print("Step: ", "%04d" % step, ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    
    print("Optimizer Finished.")
    print("Test Accuracy: ", sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))