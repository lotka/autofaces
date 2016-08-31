import sys

def crossval(s, num_folds):
    sorts = sorted(s)
    folds = [ set(sorts[i::9]) for i in range(num_folds)]
    return folds

def get_path_multi_os(dict_path):
    for key, value in dict_path.items():
        if sys.platform.startswith(key):
            return value
    raise EnvironmentError('Unknown platform: ' + sys.platform)

def binarize1_5(x):
    return x > 1.5