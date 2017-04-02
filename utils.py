# some util functions that are only for helping purposes.
import numpy as np
from sklearn.externals import joblib

def save_model(model, path_prefix):
    pca, classifier = model
    joblib.dump(pca, path_prefix+'last_pca.model')
    joblib.dump(classifier, path_prefix+'last_classifier.model')
    return

def load_model(path_prefix=''):
    pca = joblib.load(path_prefix+'last_pca.model')
    svc = joblib.load(path_prefix+'last_classifier.model')
    return pca, svc



def write_log(log, params):
    log_name = params['prefix'] + 'result.log'
    log = '\n'.join(map(str, log))
    with open(log_name, 'w') as f:
        f.write(log)
    return

def params_to_prefix(params):
    prefix = '_'.join(['-'.join(map(str, params['style_index'])), params['method'], str(params['nimages'])])
    if params['use_pca']:
        prefix = prefix + '_pca'
    return prefix + '_'




def unpack_data(data):
    image_names = map(lambda n: n[0], data)
    labels = map(lambda n: n[1], data)
    return np.array(image_names), np.array(labels)



# use this to display the confusion matrix from a last_wrong log
def confusion_matrix():
    from sklearn.metrics import confusion_matrix
    with open('logs/0-1-2-3-5_svm_1000_last_wrong.txt', 'r') as f:
        entries = f.read()
    entries = entries.split('\n')
    entries = map(lambda l: l.split(' '), entries)
    entries = filter(lambda l: len(l) == 5, entries)
    trues = map(lambda n: int(n[2]), entries)
    wrongs = map(lambda n: int(n[4]), entries)
    cm = confusion_matrix(trues, wrongs)
    print cm
