# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import os


MODEL_DIR = 'imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')



def features_of_images(images, model_data):

  # Creates graph from saved GraphDef.
  with tf.device("/cpu:0"):
      create_graph()
  # tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]

  if model_data['use_cache']:
      if not os.path.exists(model_data['cache_dir']):
          os.makedirs(model_data['cache_dir'])

  # config = tf.ConfigProto(log_device_placement=True)
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    nimages = len(images)
    # print('entered with nimages', nimages)
    feature_vector_size = model_data['feature_vector_size']
    features = np.zeros((nimages, feature_vector_size))

    tensor_name = model_data['tensor_name']
    pool_tensor = sess.graph.get_tensor_by_name(tensor_name)

    for i in xrange(nimages):
        f = feature_of_image(images[i], sess, pool_tensor, model_data)
        features[i] = f

    return features


def feature_of_image(image, sess, tensor, model_data):

    if model_data['use_cache']:
        paths = image.split('/')
        feature_dir = model_data['cache_dir']+paths[-1][:-4]+'.feature'
        if os.path.exists(feature_dir):
            feature = np.load(feature_dir)
            return feature

    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    try:
        predictions = sess.run(tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        predictions = predictions.reshape(np.product(predictions.shape))
        # cacheing this feature
        if model_data['use_cache']:
            predictions.dump(feature_dir)
        return predictions

    except Exception as inst:
        print(inst)
        print('Error occurred with %d', image)
        return None


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = MODEL_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
