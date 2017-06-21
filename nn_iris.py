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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

train_x = data[0:104, 0:4].astype('f4')
train_y = one_hot(data[0:104, 4].astype(int), 3)

valid_x = data[105:127, 0:4].astype('f4')
valid_y = one_hot(data[105:127, 4].astype(int), 3)

test_x = data[127:, 0:4].astype('f4')
test_y = one_hot(data[127:, 4].astype(int), 3)

"""
print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print
"""

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
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

    if newErr - oldErr > 0.1:
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

for b, r in zip(test_y, result):
    print b, "-->", r

print "----------------------"
print "   Test completed.  "
print "----------------------"

nErr = 0

print
print "Errors: "
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        nErr += 1
        print b, "-->", r

print
print "Errors number: ", nErr, "/23 -> ", nErr*100/23, "%"

print
