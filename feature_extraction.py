import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)
# TODO: Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs. Assign the result of the softmax activation to `probs` below.
# HINT: Look at the final layer definition in alexnet.py to get an idea of what this
# should look like.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# fc8W = tf.Variable(net_data["fc8"][0])
# fc8b = tf.Variable(net_data["fc8"][1])
# logits = tf.matmul(fc7, fc8W) + fc8b
# probabilities = tf.nn.softmax(logits)

# ------------------------SOLUTION-------------

# fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
# fc8b = tf.Variable(tf.zeros(nb_classes))
# logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
# probs = tf.nn.softmax(logits)


fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))


# # Results 0
# Image 0
# No entry: 0.053
# Keep right: 0.048
# Stop: 0.041
# Right-of-way at the next intersection: 0.040
# Go straight or right: 0.040
#
# Image 1
# Stop: 0.069
# Dangerous curve to the right: 0.059
# Yield: 0.059
# Keep right: 0.047
# No vechiles: 0.044
#
# # Results 1
# Image 0
# Right-of-way at the next intersection: 0.073
# Go straight or right: 0.063
# Speed limit (70km/h): 0.056
# Wild animals crossing: 0.048
# Keep left: 0.040
#
# Image 1
# Dangerous curve to the right: 0.089
# Speed limit (30km/h): 0.065
# Right-of-way at the next intersection: 0.061
# Speed limit (70km/h): 0.048
# Keep left: 0.048


# First, I figure out the shape of the final fully connected layer, in my opinion
# this is the trickiest part. To do that I have to figure out the size of the output
# from fc7. Since it's a fully connected layer I know it's shape will be 2D so the
# second (or last) element of the list will be the size of the output. fc7.get_
# shape().as_list()[-1] does the trick. I then combine this with the number of
# classes for the Traffic Sign dataset to get the shape of the final fully
# connected layer, shape = (fc7.get_shape().as_list()[-1], nb_classes).
# The rest of the code is just the standard way to define a fully connected
# in TensorFlow. Finally, I calculate the probabilities via softmax,
# probs = tf.nn.softmax(logits).
