
from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:\\Projects\Python\\TensorflowExamples\\', one_hot=True)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("num_steps", 200, "Number steps.")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("display_step", 10, "Display steps.")
FLAGS = tf.app.flags.FLAGS

# Paramters
num_input = 784
num_classes = 10
drop_out = 0.7   # Dropout, probability to keep unit


# Graph
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & biases
weights = {
    # 5 * 5 conv, 1 input, 32 outputs
    'wc1' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5 * 5 conv, 32 input, 64 output
    'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7 * 7 * 64 inputs, 1024 outputs
    'wd1' : tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs
    'out' : tf.Variable(tf.random_normal([1024, num_classes]))
}
bisas = {
    'bc1' : tf.Variable(tf.random_normal([32])),
    'bc2' : tf.Variable(tf.random_normal([64])),
    'bd1' : tf.Variable(tf.random_normal([1024])),
    'out' : tf.Variable(tf.random_normal([10]))
}

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution layer 1, 28 * 28 * 1 => 28 * 28 * 32
    conv1 = conv2d(x, weights['wc1'], bisas['bc1'])
    # Maxpooling layer 1, 28 * 28 * 32 => 14 * 14 * 32
    maxpool1 = maxpool2d(conv1)

    # Convolution layer 2, 14 * 14 * 32 => 14 * 14 * 64
    conv2 = conv2d(maxpool1, weights['wc2'], bisas['bc2'])
    # maxpooling layer 2, 14 * 14 * 64 => 7 * 7 * 64
    maxpool2 = maxpool2d(conv2)

    # Fully connected layer
    fc1 = tf.reshape(maxpool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, drop_out)

    # Output layer
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


logits = conv_net(X, weights, bisas, keep_prob)
pred = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for step in range(1, FLAGS.num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})
        if step % FLAGS.display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X:batch_x, Y:batch_y, keep_prob : 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    print("Optimizer finished.")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256], keep_prob: 1.0}))