import disfa2
from pyexp.pyexp import PyExp

config =  PyExp(config_file='config/cnn.yaml',path='test_disfa')
data = disfa2.Disfa(config['data'])
