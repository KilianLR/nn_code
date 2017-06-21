import gzip
import cPickle

import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y.astype(int), 10)
valid_y = one_hot(valid_y.astype(int), 10)
test_y = one_hot(test_y.astype(int), 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 15)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(15)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(15, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
oldErr = 10000
newErr = 9999
epoch = 0
stop = 0

while newErr > 1.25:
    epoch += 1

    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    if epoch > 75:
        oldErr = newErr

    newErr = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})

    print "Epoch #:", epoch, "Error: ", newErr

    if newErr - oldErr > 1:
        break;

    if newErr >= oldErr:
        if stop == 0:
            stop = 1
        else:
            break
    else:
        stop = 0

	"""
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
	"""

print "----------------------"
print "  Training completed. "
print "----------------------"
print
print "----------------------"
print "     Start test...    "
print "----------------------"

result = sess.run(y, feed_dict={x: test_x})

for i in range(10):
    print test_y[i], "-->"
    for j in range(len(result[i])):
        print "%.3f" % result[i][j],
    print

"""
for b, r in zip(test_y, result):
    print b, "-->", r
"""

print "----------------------"
print "    Test completed.   "
print "----------------------"

nErr = 0

print
print "Errors: "
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        nErr += 1
        print b, "-->", r

print
print "Errors number: ", nErr, "/10000 -> ", nErr*100/10000, "%"

print

"""
# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]


# TODO: the neural net!!
"""
