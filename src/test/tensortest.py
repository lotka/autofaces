from model import *
from src.expman import PyExp

config = PyExp(config_file='config/cnn.yaml', make_new=False)
config['data']['image_shape'] = [47,47]
config['data']['label_size'] = 12
cnn(config)