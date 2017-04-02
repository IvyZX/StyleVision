import inception
import dataset
import classify
from utils import *

import numpy as np
import cv2

import os
import datetime
import copy
import sys
import argparse


FLAGS = None

DEFAULT_PARAMS = {
    'style_index': [0, 1, 2, 3, 5],
    'method': 'svm',
    'use_pca': False,
    'nimages': 1000,
}
DEFAULT_PARAMS['style_names'] = dataset.get_style_names(DEFAULT_PARAMS['style_index'])
DEFAULT_PARAMS['prefix'] = params_to_prefix(DEFAULT_PARAMS)



def classify_image(model, path, params):
    features = classify.load_feature_vectors([path], prefix='')
    pca, classifier = model
    if params['use_pca']:
        features = pca.transform(features)
    predicted = classifier.predict(features)[0]
    predicted_style = params['style_names'][predicted]
    print 'The predicted style is {}'.format(predicted_style)
    return


def main():
    # initializing the inception part
    inception.maybe_download_and_extract()
    classify.TF_MODEL_DATA['use_cache'] = False

    # customize the parameters for this run
    params = {
        'style_index': [0, 1, 2, 3, 5],
        'method': 'svm',
        'use_pca': False,
        'nimages': 1000,
    }
    params['style_names'] = dataset.get_style_names(params['style_index'])
    params['model_prefix'] = 'models/' + params_to_prefix(params)

    print 'Candidate Artwork styles:', ', '.join(params['style_names'])
    model = load_model(params['model_prefix'])
    classify_image(model, FLAGS.image_file, params)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
      '--inception_model_dir',
      type=str,
      default='imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    inception.MODEL_DIR = FLAGS.inception_model_dir
    main()
