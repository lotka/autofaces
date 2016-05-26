import os
import yaml
from time import gmtime, strftime
"""
pyexp
"""

class PyExp:
    def __init__(self,config=None,config_file=None,path = 'data'):
        print 'Setting up folder structure.'

        """
        Configure config files
        """

        # XOR between config and config_file
        assert config != None or config_file != None
        assert config == None or config_file == None

        # Load config dictionary
        if config != None:
            self.config = config
        # Load yaml config file
        if config_file != None:
            with open(config_file, 'r') as stream:
                try:
                    self.config = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print exc

        """
        Setup folder structures
        """

        ts = strftime("%Y_%m_%d", gmtime())
        if not os.path.isdir(path):
            os.mkdir(path)

        path = os.path.join(path,ts)
        if not os.path.isdir(path):
            os.mkdir(path)

        def prefix(i):
            assert i < 1000

            if i < 10:
                return '00' + str(i)
            if i < 100:
                return '0' + str(i)
            else:
                return str(i)


        i = 1
        self.exp_path = os.path.join(path,prefix(i))
        while os.path.isdir(self.exp_path):
            i += 1
            self.exp_path = os.path.join(path,prefix(i))

        os.mkdir(self.exp_path)
        self.save_config()

    def __getitem__(self,key):
        # if type(key) == int:
        #     return self.config['experiments'][key]
        if key in self.config['global']:
            return self.config['global'][key]
        else:
            return self.config[key]


    def save_config(self):
        print 'Saving config file to', self.exp_path
        with open(os.path.join(self.exp_path,'config.yaml'), 'w') as outfile:
            outfile.write(yaml.dump(self.config, default_flow_style=False))
            outfile.close()