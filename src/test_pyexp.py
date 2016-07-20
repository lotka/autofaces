from src.pyexp import PyExp

experiment =  PyExp(config_file='config/cnn.yaml',path='test_data')
experiment.config['fuck'] = 4.0
experiment.finished()
