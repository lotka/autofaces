import numpy as np


class Analysis:
    """
    General analysis class which can either collect data during a run or take a
    dictionary describing a previous run and offer its plotting and metric functions
    """
    def __init__(self,old_dictionary=None,tf_sess=None):
        # Juist Initialize data
        if old_dictionary is None:
            pass
        else:
            pass

    def add_cost(self,name,value):
        pass

    def get_metrics(self):
        pass

    def plot_autoencoder(self):
        pass

    def plot_costs(self):
        pass

    def archive(self):
        pass
