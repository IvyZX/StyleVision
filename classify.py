#
import inception
import dataset
from utils import *

from sklearn import svm
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import cv2

import os
import datetime
import copy
import sys
import argparse


FLAGS = None

MIXED_9_MODEL = {
    'cache_dir': 'mixed_9_features/',
    'feature_vector_size': 2048*8*8,
    'tensor_name': 'mixed_9/join:0',
    }

POOL_3_MODEL = {
    'cache_dir': 'pool_3_features/',
    'feature_vector_size': 2048,
    'tensor_name': 'pool_3:0',
    }

TF_MODEL_DATA = copy.deepcopy(MIXED_9_MODEL)
TF_MODEL_DATA['use_cache'] = True

PCA_N = TF_MODEL_DATA['feature_vector_size'] / 100
RFC_N = 100


def images_to_color_bins(images):
    n = len(images)
    nbins = 100
    features = np.zeros((n, nbins*3))
    for i in xrange(n):
        image = images[i]
        area = float(image.shape[0] * image.shape[1])
        for c in xrange(3):
            hist = cv2.calcHist([image], [c], None, [nbins], [0, 256]).reshape(nbins)
            features[i, nbins*c:nbins*(c+1)] = hist / area
    return features


def load_feature_vectors(image_names, prefix=dataset.IMG_DIR, more_features=None):

    nimages = len(image_names)
    image_paths = map(lambda n: prefix+n, image_names)
    features = inception.features_of_images(image_paths, TF_MODEL_DATA)

    # add other features
    if more_features:
        features = ([features])
        images = map(cv2.imread, image_paths)
        if 'color_histogram' in more_features:
            features.append(images_to_color_bins(images))
        features = np.concatenate(features, axis=1)

    return features

def train_and_test(data, params, cv_fold=10):
    # extract features of every images
    fold_unit = len(data) / cv_fold
    np.random.shuffle(data)
    accu_rates = []
    models = []
    for fold in xrange(cv_fold):              ### only one trial for now
        print 'start fold:', fold
        train_data = data[:fold_unit*fold] + data[fold_unit*(fold+1):]
        test_data = data[fold_unit*fold:fold_unit*(fold+1)]
        model = train(train_data, params['method'], use_pca=params['use_pca'])
        print 'training done. start testing...'
        accu_rate, wrongs = test(model, test_data, use_pca=params['use_pca'])
        accu_rates.append(accu_rate)
        models.append(model)
        with open(params['prefix']+'last_wrong.txt', 'a') as log:
            for w in wrongs:
                log.write('{} truly {} classified {}\n'.format(w[0], w[1], w[2]))
    # cache the best model
    best = models[np.argmax(accu_rates)]
    save_model(best, params['prefix'])
    # write log
    log = [accu_rates, 'average: {}'.format(np.average(accu_rates)), params]
    return log


def train(data, method, use_pca=False):
    image_names, labels = unpack_data(data)
    features = load_feature_vectors(image_names)
    print 'train feature vector dim:', features.shape

    pca = decomposition.PCA(n_components=PCA_N)
    if use_pca:
        features = pca.fit_transform(features)
        print 'pca fitting done.'

    svc = svm.LinearSVC()
    rfc = RandomForestClassifier(n_estimators=RFC_N)
    abc = AdaBoostClassifier(n_estimators=100)
    nbayes = GaussianNB()

    if method == 'svm':
        classifier = svc
    if method == 'rfc':
        classifier = rfc
    if method == 'abc':
        classifier = abc
    if method == 'nbayes':
        classifier = nbayes

    classifier.fit(features, labels)
    return pca, classifier


def test(model, data, use_pca=False):
    image_names, labels = unpack_data(data)
    test_size = image_names.shape[0]
    features = load_feature_vectors(image_names)

    pca, classifier = model

    if use_pca:
        features = pca.transform(features)
    predicted = classifier.predict(features)

    accuracy = (predicted == labels)
    accu_rate = np.sum(accuracy) / float(test_size)

    print np.sum(accuracy), 'correct out of', test_size
    print 'accuracy rate: ', accu_rate

    wrongs = np.array([image_names, labels, predicted])
    wrongs = np.transpose(wrongs)[np.invert(accuracy)]

    return accu_rate, wrongs



def batch_train_test_trials():
    every_style = [0, 1, 2, 3, 5]
    # pairs = [[i, j] for i in every_style for j in every_style]
    # pairs = filter(lambda a: a[0] < a[1], pairs)
    pairs = [[1, 2]]
    # pairs = [[0, 2], [0, 5], [2, 3], [2, 5], [3, 5]]
    for pair in pairs:
        for method in ['svm']:
            params = {
                'style_index': pair,
                'method': method,
                'use_pca': False,
                'nimages': 1000,
            }
            params['prefix'] = 'logs/' + params_to_prefix(params)
            print 'prefix:', params['prefix']

            data, style_names = dataset.read_data(params['nimages'], params['style_index'])
            log = train_and_test(data, params)

            log.append('style names: {}'.format(style_names))
            log.append('nimage per style: {}'.format(params['nimages']))
            for l in log:
                print l
            write_log(log, params)



def main():
    params = DEFAULT_PARAMS
    print 'Candidate Artwork styles:', params['style_names']
    model = load_model(params['prefix'])
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
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
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
    inception.MODEL_DIR = FLAGS.model_dir
    main()
