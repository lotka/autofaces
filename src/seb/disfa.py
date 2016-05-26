import numpy as np
import data_array as data_array
import python_helper as pyh
import socket


"""
Example calls:
labels = disfa.disfa.get_array('AUall', 0)
should be the same as:
labels = disfa.disfa['AUall'][0]

Load labels:
labels_val = disfa.disfa.get_array('AUall', 0)[:]

ToDo:
- implement vertical concatenation:
  labels = disfa.disfa['AUall'][0:4]
  should return a concatenation of the first 4 subjects (without actually loading the data; a further [:] should load it

- implement image reader:
  read all images from folder in array-like structure

"""


disfa = {}
disfa_id_subj_all = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32}
disfa_basedir = {}
disfa_basedir['win'] = ['\\\\fs-vol-hmi.doc.ic.ac.uk\\hmi\\projects\\sebastian\\DISFA']
disfa_basedir['darwin'] = ['/Users/sebbi/Google Drive/Research/data/DISFA']
if socket.gethostname() == 'ux305':
    disfa_basedir['linux'] = ['/home/luka/Documents/DISFA']
else:
    disfa_basedir['linux'] = ['/vol/hmi/projects/sebastian/DISFA']
disfa_basedir = pyh.get_path_multi_os(disfa_basedir)

s = data_array.FileArray()
s.parts_path_to_file = disfa_basedir + ['Labels', 'AUall' , 'SN{:03}_labels_AUall.mat']
s.name_item = 'labels'
s.ndims = 2
s.front = -1
s.reverse = False
s.K = 6
disfa['AUall'] = s

s = data_array.FileArray()
s.parts_path_to_file = disfa_basedir + ['Features', 'points', 'SN{:03}_feature_points.mat']
s.name_item = 'coord_norm'
s.ndims = 3
s.front = 0
s.reverse = True
s.K = 1
disfa['feature_points'] = s

s = data_array.FileArray()
s.parts_path_to_file = disfa_basedir + ['Tracking', 'SN{:03}_coord.mat']
s.name_item = 'coord'
s.ndims = 3
s.front = -1
s.reverse = True
s.K = 1
disfa['coord'] = s

s = data_array.FileArray()
s.parts_path_to_file = disfa_basedir + ['Features', 'images', 'SN{:03}_features_images.mat']
s.name_item = 'images'
s.ndims = 3
s.front = 0
s.reverse = True
s.K = 1
disfa['images'] = s

disfa_ic_all = [(i, np.s_[:]) for i in sorted(disfa_id_subj_all)]
# example how to get all 'coord':
# coord, id_array = data_array.IndicesCollection(disfa.disfa_ic_all).getitem(disfa.disfa['coord'])
