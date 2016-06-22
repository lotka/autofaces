import ruamel.yaml as yaml
print 'wtf'
import os

from time import gmtime, strftime
import subprocess
"""
pyexp
"""

class PyExp:
    def __init__(self,config=None,config_file=None,path = 'data',make_new=True):

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
                    self.config = yaml.load(stream,yaml.RoundTripLoader)
                except yaml.YAMLError as exc:
                    print exc
        if make_new:
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
            os.mkdir(os.path.join(self.exp_path,'images'))

            # Save data path
            self.config['data']['path'] = path

            """
            Get git commit
            """
            label = subprocess.check_output(["git", "describe","--always"]).rstrip()
            self.config['global']['commit'] = label
            self.save_config()

            """
            Write NOT_FINISHED file
            """
            self.not_finished_file = os.path.join(self.exp_path,'NOT_FINISHED')
            print 'OPENING ', self.not_finished_file
            f = open(self.not_finished_file,'w')
            f.write('THIS FILE MEANS THIS RUN DID NOT REACH COMPLETION')
            f.close()


    def __getitem__(self,key):
        # if type(key) == int:
        #     return self.config['experiments'][key]
        if key in self.config['global']:
            return self.config['global'][key]
        else:
            return self.config[key]

    def update(self,key,new_val,save=True):
        self.config[key]=new_val
        if save:
            self.save_config()

    def get_path(self):
        return self.exp_path


    def save_config(self):
        print 'Saving config file to', self.exp_path
        with open(os.path.join(self.exp_path,'config.yaml'), 'w') as outfile:
            outfile.write(yaml.dump(self.config, default_flow_style=False,Dumper=yaml.RoundTripDumper))
            outfile.close()

    def finished(self):
        self.save_config()
        os.remove(self.not_finished_file)
        finished_file = os.path.join(self.exp_path,'FINISHED')
        f = open(finished_file,'w')
        f.write('RUN FINISHED')
        f.close()
        print 'Finished run with path: '
        print self.exp_path
