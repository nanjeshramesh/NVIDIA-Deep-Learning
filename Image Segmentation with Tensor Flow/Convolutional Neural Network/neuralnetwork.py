#
# Copyright 2017 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import tensorflow as tf

# function to print the tensor shape.  useful for debugging

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())


def read_and_decode(filename_queue):

# input: filename
# output: image, label pair

# setup a TF record reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

# list the features we want to extract, i.e., the image and the label
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

  # Decode the training image
  # Convert from a scalar string tensor (whose single string has
  # length 256*256) to a float tensor
    image = tf.decode_raw(features['img_raw'], tf.int64)
    image.set_shape([65536])
    image_re = tf.reshape(image, (256,256))

# Scale input pixels by 1024
    image_re = tf.cast(image_re, tf.float32) * (1. / 1024)

# decode the label image, an image with all 0's except 1's where the left
# ventricle exists
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape([65536])
    label_re = tf.reshape(label, [256,256])

    return image_re, label_re


def inputs(batch_size, num_epochs, filename):

# inputs: batch_size, num_epochs are scalars, filename
# output: image and label pairs for use in training or eval

    if not num_epochs: num_epochs = None

# define the input node
    with tf.name_scope('input'):

# setup a TF filename_queue
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

# return and image and label
        image, label = read_and_decode(filename_queue)
     
# shuffle the images, not strictly necessary as the data creating
# phase already did it, but there's no harm doing it again.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=15,
            min_after_dequeue = 10)

#        tf.image_summary( 'images', tf.reshape(images,[-1,256,256,1] ))
#        tf.image_summary( 'labels', tf.reshape(sparse_labels,[-1,256,256,1]))
        return images, sparse_labels


def inference(images):

#   input: tensor of images
#   output: tensor of computed logits

    print_tensor_shape( images, 'images shape inference' )

# resize the image tensors to add the number of channels, 1 in this case
# required to pass the images to various layers upcoming in the graph
    images_re = tf.reshape( images, [-1,256,256,1] ) 
    print_tensor_shape( images, 'images shape inference' )

# Convolution layer

# Using TensorFlow layers API is a higher level interface than creating
# weight and bias tensors and combining them with conv operations.  The Conv2d 
# performs a 2d convolution with bias and activation built-in.

# filters -- number of feature maps to be output
# kernel_size -- the size of the feature maps
# kernel_initializer -- sets up the initial weights of the feature maps
# strides -- stride of the sliding window

    conv1 = tf.layers.conv2d( images_re, 
        filters=100, 
        kernel_size=[5,5], 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        strides=[2,2], 
        activation=tf.nn.relu, 
        use_bias=False, 
        padding='same', 
        name='Conv1')

    print_tensor_shape( conv1, 'conv1 shape')

# Pooling layer

# TensorFlow pooling layer includes the size of the sliding window and the 
# stride of the window

    pool1 = tf.layers.max_pooling2d( conv1, 
        pool_size=[2,2], 
        strides=[2,2], 
        padding='same', 
        name='Pool1')

    print_tensor_shape( pool1, 'pool1 shape')

# Conv layer

    conv2 = tf.layers.conv2d( pool1, 
        filters=200, 
        kernel_size=[5,5], 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        strides=[2,2], 
        activation=tf.nn.relu, 
        use_bias=False, 
        padding='same',
        name='Conv2' )

    print_tensor_shape( conv2, 'conv2 shape')

# Pooling layer

    pool2 = tf.layers.max_pooling2d( conv2, 
        pool_size=[2,2], 
        strides=[2,2], 
        padding='same', 
        name='Pool2')

    print_tensor_shape( pool2, 'pool2 shape')
    
# Conv layer

    conv3 = tf.layers.conv2d( pool2, 
        filters=300, 
        kernel_size=[3,3], 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        strides=[1,1], 
        activation=tf.nn.relu, 
        use_bias=False, 
        padding='same',
        name='Conv3' )

    print_tensor_shape( conv3, 'conv3 shape')
    
# Conv layer

    conv4 = tf.layers.conv2d( conv3, 
        filters=300, 
        kernel_size=[3,3],
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        strides=[1,1], 
        activation=tf.nn.relu, 
        use_bias=False, 
        padding='same', 
        name='conv4')

    print_tensor_shape( conv4, 'conv4 shape')

    drop = tf.nn.dropout( conv4, 1.0 )
    print_tensor_shape( drop, 'drop shape' )
    
# Conv layer to generate the 2 score classes

    score_classes = tf.layers.conv2d( drop,
        filters=2,
        kernel_size=[1,1], 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        strides=[1,1], 
        use_bias=False,
        padding='same', 
        name='score_classes')

    print_tensor_shape( score_classes,'score_classes shape')

# Upscore the results to 256x256x2 image

    upscore = tf.layers.conv2d_transpose( score_classes,
        filters=2, 
        kernel_size=[31,31], 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        strides=[16,16], 
        use_bias=False, 
        padding='same', 
        name='upscore')

    print_tensor_shape(upscore, 'upscore shape')

    return upscore

def loss(logits, labels):
    
    # input: logits: Logits tensor, float - [batch_size, 256, 256, NUM_CLASSES].
    # intput: labels: Labels tensor, int32 - [batch_size, 256, 256].
    # output: loss: Loss tensor of type float.

    labels = tf.to_int64(labels)
    print_tensor_shape( logits, 'logits shape before')
    print_tensor_shape( labels, 'labels shape before')

# reshape to match args required for the cross entropy function
    logits_re = tf.reshape( logits, [-1, 2] )
    labels_re = tf.reshape( labels, [-1] )
    print_tensor_shape( logits, 'logits shape after')
    print_tensor_shape( labels, 'labels shape after')

# call cross entropy with logits
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
         labels=labels, logits=logits, name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy, name='1cnn_cross_entropy_mean')
    return loss


def training(loss, learning_rate, decay_steps, decay_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(loss.op.name, loss)

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

  # create learning_decay
    lr = tf.train.exponential_decay( learning_rate,
                                     global_step,
                                     decay_steps,
                                     decay_rate, staircase=True )

    tf.summary.scalar('1learning_rate', lr )

  # Create the gradient descent optimizer with the given learning rate.
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(lr)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(logits, labels):
    # input: logits: Logits tensor, float - [batch_size, 256, 256, NUM_CLASSES].
    # input: labels: Labels tensor, int32 - [batch_size, 256, 256]
    # output: scaler int32 tensor with number of examples that were 
    #         predicted correctly

    with tf.name_scope('eval'):
        labels = tf.to_int64(labels)
        print_tensor_shape( logits, 'logits eval shape before')
        print_tensor_shape( labels, 'labels eval shape before')

# reshape to match args required for the cross entropy function
        logits_re = tf.reshape( logits, [-1, 2] )
        labels_re = tf.reshape( labels, [-1] )
        print_tensor_shape( logits, 'logits eval shape after')
        print_tensor_shape( labels, 'labels eval shape after')

  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
        correct = tf.nn.in_top_k(logits_re, labels_re, 1)
        print_tensor_shape( correct, 'correct shape')

  # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))
