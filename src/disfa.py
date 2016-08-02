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
        new_image = scipy.misc.imresize(new_image,[int(new_image.shape[0]*scale),int(new_image.shape[1]*scale)])

        return new_image

    def inverse_process(self,input,base=0):

        N = input.shape[0]
        output = input.copy()
        if self.config['scaling'] == 'maxdiv':
            output = self.normalise_range(output,inverse=True)
        if self.config['scaling'] == '[-1,1]':
            output = self.scale_within_range(output, inverse=True)
        # elif self.config['scaling'] == '[0,1]':
        #         output = output*self.max + self.min

        if self.config['normalisation_type'] == 'pixel':
            output = output*self.std + self.mean
        elif self.config['normalisation_type'] == 'face':
            for i in tqdm(xrange(N)):
                output[i,:,:] = output[i,:,:]*self.std + self.mean
        elif self.config['normalisation_type'] == 'contrast':
            for i in tqdm(xrange(N)):
                output[i,:,:] *= self.std[i+base]
                output[i,:,:] += self.mean[i+base]

        return output

    def mean_squared(self,x,y):
        return np.sqrt((np.power(x-y,2)).mean())

    def true_loss(self,input,output,base=0):
        assert input.shape == output.shape
        true_input = self.inverse_process(input,base)
        true_output = self.inverse_process(output,base)
        N, x, y = input.shape
        losses = np.zeros(N)
        for i in xrange(N):
            losses[i] = self.mean_squared(true_input[i,:,:],true_output[i,:,:])
        return losses

    def zeros_to_ones(self,array):
        return array + (array == 0.0).astype(array.dtype)

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

    def scale_within_range(self, x,inverse=False):
        assert len(x.shape) == 3
        y = np.zeros(x.shape)
        N = x.shape[0]
        if not inverse:
            self.min = x.min(axis=0)
            self.max = x.max(axis=0)

        _range = self.max - self.min
        sub = (self.min + _range / 2.0)
        div = self.zeros_to_ones((_range / 2.0))

        for i in xrange(N):
            if not inverse:
                y[i] = (x[i] - sub) / div
            if inverse:
                y[i] = (x[i] * div) + sub
        return y

    def normalise(self,images):
        option = self.config['normalisation_type']
        print option
        N = images.shape[0]

        if option == 'pixel':
            self.mean =  images.mean()
            self.std = images.std()
            images =  (images - self.mean)/self.std
        elif option == 'face':
            mean_face = images.mean(axis=0)
            stdd_face = self.zeros_to_ones(images.std(axis=0))

            self.mean =  mean_face
            self.std = stdd_face

            for i in tqdm(xrange(N)):
                images[i,:,:] = (images[i,:,:] - self.mean)/self.std
        elif option == 'contrast':
            self.mean = np.ones(N)
            self.std  = np.ones(N)
            for i in tqdm(xrange(N)):
                self.mean[i] = images[i,:,:].mean()
                # if np.abs(self.std[i]) < 0.000001:
                #     self.mean[i] = 0.0
                self.std[i] = images[i,:,:].std()
                if np.abs(self.std[i]) == 0.0:
                    self.std[i] = 1.0

            self.mean[i] = np.random.rand()
            self.std[i] = np.random.rand()
            for i in tqdm(xrange(N)):
                images[i,:,:] -= self.mean[i]
                images[i,:,:] /= self.std[i]


        print 'Applying scaling: ', self.config['scaling']
        if self.config['scaling'] == 'maxdiv':
            images = self.normalise_range(images)
        if self.config['scaling'] == '[-1,1]':
            images = self.scale_within_range(images)
        # elif self.config['scaling'] == '[0,1]':
        #     self.min = images[:, :, :].min()
        #     self.max = images[:, :, :].max()
        #     images = (images - self.min)/self.max

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

        # Apply threshold
        lx,ly = self.labels.shape
        for i in xrange(lx):
            for j in xrange(ly):
                if self.labels[i,j] >= self.config['threshold']:
                    self.labels[i,j] = 1.0
                else:
                    self.labels[i,j] = 0.0

        self.images = self.images/self.images.max()
        self.images = self.normalise(self.images)

        # Remove the empty labels
        if self.config['remove_empty_labels'] and self.batch_type == 'train':
            good_samples = []
            for i in xrange(self.images.shape[0]):
                if self.labels[i,:].sum() > 0.0:
                    good_samples.append(i)

            self.images = self.images[good_samples,:,:]
            self.labels = self.labels[good_samples,:]

            self.N = self.images.shape[0]
            assert self.images.shape[0] == self.labels.shape[0]



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


    def __init__(self,
                 config,
                 batch_type):

        self.batch_counter = 0
        # copy ensures accurate hashing later on
        self.config = config
        self.batch_type = batch_type
        subjects = config[batch_type + '_subjects']
        self.ranges = None
        self.min = None
        self.max = None
        self.mean = None
        self.std = None
        self.absmax = None
        self.mean_image = None

        print 'Creating',batch_type,'batch.'
        print 'Loading labels and images for subjects:',subjects
        subjects_set = [(i, np.s_[:]) for i in sorted(set(subjects))]
        self.labels, _ = data_array.IndicesCollection(subjects_set).getitem(disfa.disfa['AUall'])

        # Save original label ordering
        self.raw_labels = self.labels.copy()

        self.image_region_config = self.config[self.config['image_region']]

        if 'normalisation' in config:
            if config['normalisation'] == 'none_[0,1]':
                config['normalisation_type'] = 'none'
                config['scaling'] = 'none'

            elif config['normalisation'] == 'none_[-1,1]':
                config['normalisation_type'] = 'none'
                config['scaling'] = '[-1,1]'

            elif config['normalisation'] == 'face_[-inf,inf]':
                config['normalisation_type'] = 'face'
                config['scaling'] = 'none'

            elif config['normalisation'] == 'face_[-1,1]':
                config['normalisation_type'] = 'face'
                config['scaling'] = '[-1,1]'

            elif config['normalisation'] == 'contrast_[-inf,inf]':
                config['normalisation_type'] = 'contrast'
                config['scaling'] = 'none'

            elif config['normalisation'] == 'contrast_[-1,1]':
                config['normalisation_type'] = 'contrast'
                config['scaling'] = '[-1,1]'

        def hash_config(conf):
            _conf = copy(conf)
            ignore_for_hashing = ['path','image_shape','label_size','results']
            for name in ignore_for_hashing:
                if name in _conf:
                    del _conf[name]
            return hash_anything(_conf)

        hashing_enabled = True

        if hashing_enabled:

            h1 = hash_config(self.config)
            h2 = hash_config(subjects)
            h3 = 0

            with open('disfa.py','r') as file:
                h3 = hash(file.read())

            h = h1 ^ h2 ^ h3
            hash_folder = join(join(config['path'],'..'),'hashed_datasets')
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
            self.nSamples = self.images.shape[0]
            self.labels = d['labels']
            self.min = d['min']
            self.max = d['max']
            self.absmax = d['absmax']
            self.mean = d['mean']
            self.std = d['std']
            self.ranges = d['ranges']
            self.mean_image = d['mean_image']
            assert (self.images.sum() + self.labels.sum()) == d['checksum']
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
                self.images = np.append(self.images,r_images,axis=0)

                del r_images
            self.nSamples = self.images.shape[0]
            if self.config['batch_randomisation']:
                if batch_type == 'train':
                    # Generate some random indices
                    np.random.seed(0)
                    idx = np.random.choice(self.nSamples, self.nSamples, replace=False)
                    # Shuffle the train and validation set with the indicies
                    self.images = self.images[idx, :, :]
                    self.labels = self.labels[idx, :]

            self.mean_image = self.images.mean(axis=0)
            # Finally perform processing on all of the images as a whole
            self.all_images_pre_process()

            if hashing_enabled:
                print 'SAVING HASHFILE FOR FUTURE USE', batch_type
                checksum = self.images.sum() + self.labels.sum()
                np.savez(hash_file,images=self.images,
                                   labels=self.labels,
                                   min=self.min,
                                   max=self.max,
                                   mean=self.mean,
                                   absmax=self.absmax,
                                   mean_image=self.mean_image,
                                   std=self.std,
                                   ranges=self.ranges,
                                   checksum=checksum)




class Disfa:
    def __init__(self,config):


        # self.test       = Batch(config,'test')
        self.train      = Batch(config,'train')
        self.validation = Batch(config,'validation')

        self.config = config
        self.config['image_shape'] = [self.train.images.shape[1],self.train.images.shape[2]]
        self.config['label_size'] = self.train.labels.shape[1]
