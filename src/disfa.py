import numpy as np
import seb.data_array as data_array
import seb.disfa as disfa
from scipy import ndimage
import scipy
from tqdm import tqdm
from copy import copy
from os.path import join
import time
import os
from helper import hash_anything

class Batch:


    def __init__(self,
                 config,
                 batch_type,
                 preset_faces=None,
                 debug=False):


        def noneSet(original_value,new_value):
            if new_value is None:
                return original_value
            else:
                return new_value

        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.contrast_mean = None
        self.contrast_std = None

        if preset_faces != None:
            self.min  = noneSet(self.min,  preset_faces['min'])
            self.max  = noneSet(self.max,  preset_faces['max'])
            self.mean = noneSet(self.mean, preset_faces['mean'])
            self.std  = noneSet(self.std,  preset_faces['std'])


        self.batch_counter = 0
        # copy ensures accurate hashing later on
        self.config = config
        self.batch_type = batch_type
        subjects = config[batch_type + '_subjects']
        self.absmax = None
        self.mean_image = None
        self.subject_idx = []
        self.debug = debug
        self.ranges = None

        if 'preprocessing' not in self.config:
            p = {'contrast'    : False,
                 'face'        : False,
                 'per_subject' : False,
                 'range'       : False}
        if 'normalisation_type' in self.config:
            if self.config['normalisation_type'] == 'face_ps' or self.config['normalisation_type'] == 'face':
                p['face']     = True
            if self.config['normalisation_type'] == 'contrast':
                p['contrast'] = True
        if 'scaling' in self.config:
            if self.config['scaling'] == '[-1,1]':
                p['range'] = True

            self.config['preprocessing'] = p

        self.options = self.config['preprocessing']

        print 'Creating',batch_type,'batch.'
        print 'Loading labels and images for subjects:',subjects
        subjects_set = [(i, np.s_[:]) for i in sorted(set(subjects))]
        self.labels, _ = data_array.IndicesCollection(subjects_set).getitem(disfa.disfa['AUall'])

        # Save original label ordering
        self.raw_labels = self.labels.copy()

        self.image_region_config = self.config[self.config['image_region']]

        def hash_config(conf):
            _conf = copy(conf)
            ignore_for_hashing = ['path','image_shape','label_size','results']
            for name in ignore_for_hashing:
                if name in _conf:
                    del _conf[name]
            return hash_anything(_conf)

        hashing_enabled = False

        if hashing_enabled:
            h1 = hash_config(self.config)
            h2 = hash_config(subjects)
            h3 = 0

            with open('disfa.py','r') as file:
                h3 = hash(file.read())

            h = h1 ^ h2 ^ h3
            hash_folder = join(join(self.config['path'],'..'),'hashed_datasets')
            hash_file = join(hash_folder,batch_type+'_'+str(h))

            if not os.path.isdir(hash_folder):
                os.mkdir(hash_folder)

            print hash_file+'.npz'

        if hashing_enabled and os.path.isfile(hash_file+'.npz'):
            try:
                d = np.load(hash_file+'.npz')
            except IOError:
                time.sleep(60)
                d = np.load(hash_file + '.npz')

            self.images = d['images']
            self.subject_idx = d['subject_idx']
            self.nSamples = self.images.shape[0]
            self.labels = d['labels']
            self.min = d['min']
            self.max = d['max']
            self.absmax = d['absmax']
            self.mean = d['mean']
            self.std = d['std']
            self.contrast_mean = d['contrast_mean']
            self.contrast_std = d['contrast_std']
            self.ranges = d['ranges']
            self.mean_image = d['mean_image']
            assert (self.images.shape[0] + self.labels.shape[0]) == d['checksum']
            print 'LOADED FROM HASHED DATASET FILE', batch_type
        else:
            for i,s in enumerate(subjects):
                images = disfa.disfa['images'][s][:].astype(np.float32)

                # Discover dimensions of images
                if i == 0:
                    number_of_frames_per_subject = images.shape[0]
                    # Process one image to see the output
                    x,y = self.image_pre_process(images[0,:,:]).shape

                    # Set up variable for image data
                    self.images = np.empty((0,x,y))


                # Allocate memory for downsized images
                r_images = np.zeros((images.shape[0],x,y))

                # Process each image
                for i in tqdm(xrange(images.shape[0])):
                    r_images[i,:,:] = self.image_pre_process(images[i,:,:])

                # r_images = self.subject_pre_process(r_images)

                # append down_sized images to all_images
                self.subject_idx.append([s,self.images.shape[0]])
                if self.debug:
                    print 'new subject idx tuple: ', self.subject_idx[-1]
                self.images = np.append(self.images,r_images,axis=0)

                del r_images
            self.nSamples = self.images.shape[0]
            self.images_original = self.images.copy()
            self.mean_image = self.images.mean(axis=0)
            self.all_images_pre_process()
            if self.config['batch_randomisation'] and batch_type == 'train':
                # Generate some random indices
                np.random.seed(0)
                idx = np.random.choice(self.nSamples, self.nSamples, replace=False)
                # Shuffle the train and validation set with the indicies
                self.images = self.images[idx, :, :]
                self.labels = self.labels[idx, :]
            self.nSamples = self.images.shape[0]

            if hashing_enabled:
                print 'SAVING HASHFILE FOR FUTURE USE', batch_type
                checksum = self.images.shape[0] + self.labels.shape[0]
                np.savez(hash_file,images=self.images,
                                   labels=self.labels,
                                   min=self.min,
                                   max=self.max,
                                   mean=self.mean,
                                   std=self.std,
                                   contrast_mean=self.contrast_mean,
                                   contrast_std=self.contrast_std,
                                   absmax=self.absmax,
                                   mean_image=self.mean_image,
                                   ranges=self.ranges,
                                   subject_idx=self.subject_idx,
                                   checksum=checksum)



    def image_pre_process(self,image):
        # unpack required information
        lc,rc,tc,bc = self.image_region_config['crop']
        scale = self.image_region_config['resize_scale']
        x,y = image.shape

        # Calculate crop indicies
        l = int(0 + x*lc)
        r = int(x - x*rc)
        u = int(0 + y*tc)
        d = int(y - y*bc)

        # Apply crop
        new_image = image[u:d,l:r]

        # Resize the image
        new_image = scipy.misc.imresize(new_image,[int(new_image.shape[0]*scale),int(new_image.shape[1]*scale)],interp='nearest')

        return new_image

    def get_subject_id(self,s):
        if hasattr(s, '__iter__'):
            res = []
            for i in s:
                res.append(self.get_subject_id(i))
            return res
        else:
            for j in xrange(len(self.subject_idx) - 1):
                if s >= self.subject_idx[j][1] and s <= self.subject_idx[j + 1][1]:
                    return self.subject_idx[j][0]
            return self.subject_idx[-1][0]

    def get_local_subject_id(self,s):
        if type(s) == type([]):
            res = []
            for i in s:
                res.append(self.get_local_subject_id(i))
            return res
        else:
            subs = self.config[self.batch_type + '_subjects']
            for i in xrange(len(subs)):
                if subs[i] == s:
                    return i

    def inverse_process(self,input,base=0,idx=None):
        assert base == 0 or idx is None, 'inverse_process: Either base or idx should be left as default parameters.'

        if self.debug:
            print 'DEBUG:' + self.batch_type + ':',
        N, xN, yN = input.shape
        output = input.copy()

        if self.options['contrast']:
            if idx != None:
                mean = self.contrast_mean.copy()[idx]
                std = self.contrast_std.copy()[idx]
            else:
                mean = self.contrast_mean[base:]
                std = self.contrast_std[base:]

            for i in tqdm(xrange(N)):
                output[i,:,:] *= std[i]
                output[i,:,:] += mean[i]

        if self.options['range']:
            output,_,_ = self.scale_within_range(output, inverse=True)
        # elif self.config['scaling'] == '[0,1]':
        #         output = output*self.max + self.min

        if self.options['face'] and not self.options['per_subject']:

            assert self.std.shape == output.shape[1:], 'Error in face normalisation'+str(self.std.shape)+'!='+str(output.shape[1:])
            assert self.mean.shape == output.shape[1:], 'Error in face normalisation'+str(self.mean.shape)+'!='+str(output.shape[1:])

            for i in xrange(N):
                output[i,:,:] = (output[i,:,:]*self.std) + self.mean
        elif self.options['face'] and self.options['per_subject']:
            assert self.std.shape[1:] == output.shape[1:], 'Error in face normalisation'
            assert self.mean.shape[1:] == output.shape[1:], 'Error in face normalisation'
            if idx != None:
                indicies = self.get_local_subject_id(self.get_subject_id(idx))
            else:
                indicies = []
                for i in xrange(base,output.shape[0]+base):
                    indicies.append(self.get_local_subject_id(self.get_subject_id(i)))

            # assert False, str(indicies)
            for i in tqdm(xrange(output.shape[0])):
                output[i, :, :] = output[i, :, :] * self.std[indicies[i]] + self.mean[indicies[i]]

        return output

    def normalise_range(self, x,inverse=False):
        assert len(x.shape) == 3
        y = np.zeros(x.shape)
        if not inverse:
            self.absmax = self.zeros_to_ones( np.abs(x).max(axis=0) )

        for i in xrange(y.shape[0]):
            for j in xrange(y.shape[1]):
                for k in xrange(y.shape[2]):
                    if not inverse:
                        y[i, j, k] = x[i, j, k] / self.absmax[j,k]
                    else:
                        y[i, j, k] = x[i, j, k] * self.absmax[j, k]
            # y[i,:,:] = (M - m) * (x[i,:,:] - Min) /self.zeros_to_ones((Max - Min) + m)
        return y

    def scale_within_range(self, x,inverse=False,max=None,min=None):
        assert len(x.shape) == 3
        y = np.zeros(x.shape)
        N = x.shape[0]
        if not inverse and max is None and min is None:
            _min = x.min(axis=0)
            _max = x.max(axis=0)
        if max != None and min !=None:
            _min = min
            _max = max
        if inverse:
            _max = self.max
            _min = self.min

        _range = _max - _min
        sub = (_min + _range / 2.0)
        div = self.zeros_to_ones((_range / 2.0))

        for i in xrange(N):
            if not inverse:
                y[i] = (x[i] - sub) / div
            if inverse:
                y[i] = (x[i] * div) + sub
        return y, _max, _min

    def mean_face_preprocess(self, input,mean=None,std=None):
        output = input.copy()
        if mean is None and std is None:
            mean_face = output.mean(axis=0)
            stdd_face = self.zeros_to_ones(output.std(axis=0))
        else:
            mean_face = mean
            stdd_face = std

        N, _, _ = output.shape
        for i in tqdm(xrange(N)):
            output[i, :, :] = (output[i, :, :] - mean_face) / stdd_face

        return output, mean_face, stdd_face

    def apply_preprocessing(self, images):

        N, xN, yN = images.shape

        if not self.options['per_subject']:
            if self.options['face']:
                if self.mean is None and self.std is None:
                    images, self.mean, self.std = self.mean_face_preprocess(images)
                else:
                    images, self.mean, self.std = self.mean_face_preprocess(images,mean=self.mean,std=self.std)
            if self.options['range']:
                if self.max is None and self.min is None:
                    images, self.max, self.min = self.scale_within_range(images)
                else:
                    images, self.max, self.min = self.scale_within_range(images,max=self.max,min=self.min)
        elif self.options['per_subject']:
            if self.options['face']:
                self.mean = []
                self.std = []
            if self.options['range']:
                self.max = []
                self.min = []

            subs = self.subject_idx
            for j in xrange(len(self.subject_idx)):
                if j < len(self.subject_idx) - 1:
                    left = subs[j][1]; right = subs[j+1][1]
                else:
                    left = subs[j][1]; right = self.images.shape[0]
                subject_images = self.images[left:right]
                if self.options['face']:
                    self.images[left:right], m, s = self.mean_face_preprocess(subject_images)
                    self.mean.append(m); self.std.append(s)
                if self.options['range']:
                    images, _max, _min = self.scale_within_range(images)
                    self.max.append(_max); self.min.append(_min)
                if self.debug:
                    print '(left,right) = (', left, ',', right, ')'
                    print 'subject image shape is ', subject_images.shape

            if self.options['face']:
                self.mean = np.array(self.mean)
                self.std = np.array(self.std)
            if self.options['range']:
                self.max = np.array(self.max)
                self.min = np.array(self.min)
        if self.options['contrast']:
            self.contrast_mean = np.zeros(N)
            self.contrast_std  = np.zeros(N)
            print self.contrast_mean
            print self.contrast_std
            for i in tqdm(xrange(N)):
                self.contrast_mean[i] = images[i,:,:].mean()
                self.contrast_std[i] = images[i,:,:].std()
                if np.abs(self.contrast_std[i]) == 0.0:
                    self.contrast_std[i] = 1.0

                images[i,:,:] -= self.contrast_mean[i]
                images[i,:,:] /= self.contrast_std[i]

        return images

    def all_images_pre_process(self):

        # Remove undesired aus
        if 'AUs' in self.image_region_config:
            aus = self.image_region_config['AUs']
            nLabels = self.labels.shape[0]
            cropped_all_labels = np.zeros((nLabels,len(aus)))
            # self.config['sample_number'] = nLabels

            for i in xrange(nLabels):
                cropped_all_labels[i,:] = self.labels[i,aus]

            self.labels = cropped_all_labels


        self.images = self.images/self.images.max()
        self.images = self.apply_preprocessing(self.images)
        # Remove the empty labels
        good_sample_criteria = 0.0
        if 'good_sample_criteria' in self.config:
            good_sample_criteria = self.config['good_sample_criteria']
        if self.config['remove_empty_labels'] and self.batch_type == 'train':
            good_samples = []
            for i in xrange(self.images.shape[0]):
                #good_sample_criteria normally = 0 unless overwrriten in config
                if self.labels[i,:].sum() > good_sample_criteria:
                    good_samples.append(i)

            self.images = self.images[good_samples,:,:]
            self.labels = self.labels[good_samples,:]

            self.N = self.images.shape[0]
            assert self.images.shape[0] == self.labels.shape[0]

        # Apply threshold
        lx,ly = self.labels.shape
        for i in xrange(lx):
            for j in xrange(ly):
                if self.labels[i,j] >= self.config['threshold']:
                    self.labels[i,j] = 1.0
                else:
                    self.labels[i,j] = 0.0

        print self.images.shape



    def next_batch(self,n,part=0,parts=1):

        if self.batch_type == 'validation':
            left = part * self.nSamples / parts
            right = (part + 1) * self.nSamples / parts - 1
            idx = np.linspace(left,right,n,dtype=int)
            return self.images[idx,:,:],self.labels[idx,:]

        if n < 0:
            return self.images, self.labels
        else:
            # Load the indicies
            i = self.batch_counter
            N = self.nSamples
            # Increment index
            if self.batch_type != 'validation':
                self.batch_counter += n
            # Generate desired indicies, not sure if this is the best
            # way to do this in numpy but oh well
            idx = []
            for j in range(i,i+n):
                idx.append(j % N)

            # print '(',i,i+n,N,')',

            return self.images[idx,:,:],self.labels[idx,:]

    def mean_squared(self, x, y):
        return np.sqrt((np.power(x - y, 2)).mean())


    def true_loss(self, input, output, base=0):
        assert input.shape == output.shape
        true_input = self.inverse_process(input, base)
        true_output = self.inverse_process(output, base)
        N, x, y = input.shape
        losses = np.zeros(N)
        for i in xrange(N):
            losses[i] = self.mean_squared(true_input[i, :, :], true_output[i, :, :])
        return losses


    def zeros_to_ones(self, array):
        return array + (array == 0.0).astype(array.dtype)


from copy import copy

class Disfa:
    def __init__(self,config,debug=False):
        # self.test       = Batch(copy(config),'test',debug=debug)
        self.train      = Batch(copy(config),'train',debug=debug)
        p = {}
        p['min']  = self.train.min
        p['max']  = self.train.max
        p['mean'] = self.train.mean
        p['std']  = self.train.std
        self.validation = Batch(copy(config),'validation',debug=debug,preset_faces=p)

        self.config = config
        self.config['image_shape'] = [self.train.images.shape[1],self.train.images.shape[2]]
        self.config['label_size'] = self.train.labels.shape[1]
