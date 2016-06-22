import numpy as np
import seb.data_array as data_array
import seb.disfa as disfa
from scipy import ndimage
import scipy
from tqdm import tqdm
from copy import copy
from os.path import join
import os

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

    def subject_pre_process(self,images):
        option = self.config['normalisation_type']

        if option == 'pixel':
            images =  (images - np.ones(images.shape)*images.mean())/images.std()
        elif option == 'face':
            N = images.shape[0]
            mean_face = images.mean(axis=0)
            stdd_face = images.std(axis=0)
            x,y = stdd_face.shape
            for i in xrange(x):
                for j in xrange(y):
                    if stdd_face[i,j] == 0.0:
                        stdd_face[i,j] = 1.0

            for i in xrange(N):
                images[i,:,:] = (images[i,:,:] - mean_face)/stdd_face
        else:
            raise Exception("normalisation_type not set correctly: unknown type " + option)

        if self.config['normalisation_between_minus_one_and_one']:
            images = images/np.abs(images).max()

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



    def next_batch(self,n):

        if self.batch_type == 'validation':
            idx = np.linspace(0,self.nSamples-1,n,dtype=int)
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

            return self.images[idx,:,:],self.labels[idx,:]



    def __init__(self,
                 config,
                 batch_type):

        self.batch_counter = 0
        # copy ensures accurate hashing later on
        self.config = config
        self.batch_type = batch_type
        subjects = config[batch_type + '_subjects']

        print 'Creating',batch_type,'batch.'
        print 'Loading labels and images for subjects:',subjects
        subjects_set = [(i, np.s_[:]) for i in sorted(set(subjects))]
        self.labels, _ = data_array.IndicesCollection(subjects_set).getitem(disfa.disfa['AUall'])

        # Save original label ordering
        self.raw_labels = self.labels.copy()

        self.image_region_config = self.config[self.config['image_region']]

        def hash_config(conf):
            _conf = copy(conf)
            bad = ['path','image_shape','label_size']
            for x in bad:
                if x in _conf:
                    _conf[x] = '?'
            return hash(frozenset(_conf))

        h1 = hash_config(self.config)
        h2 = hash_config(subjects)
        h3 = 0

        with open('disfa.py','r') as file:
            h3 = hash(file.read())

        h = h1 ^ h2 ^ h3
        hash_folder = join(config['path'],'hashed_datasets')
        hash_file = join(hash_folder,batch_type+'_'+str(h))


        if not os.path.isdir(hash_folder):
            os.mkdir(hash_folder)

        print hash_file+'.npz'

        if os.path.isfile(hash_file+'.npz'):
            d = np.load(hash_file+'.npz')
            self.images = d['images']
            self.labels = d['labels']
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

                r_images = self.subject_pre_process(r_images)

                # append down_sized images to all_images
                self.images = np.append(self.images,r_images,axis=0)

                del r_images

            # Finally perform processing on all of the images as a whole
            self.all_images_pre_process()

            print 'SAVING HASHFILE FOR FUTURE USE', batch_type
            checksum = self.images.sum() + self.labels.sum()
            np.savez(hash_file,images=self.images,labels=self.labels,checksum=checksum)

        self.nSamples = self.images.shape[0]

        if self.config['batch_randomisation']:
            if batch_type == 'train':
                # Generate some random indices
                np.random.seed(0)
                idx = np.random.choice(self.nSamples,self.nSamples,replace=False)
                # Shuffle the train and validation set with the indicies
                self.images = self.images[idx,:,:]
                self.labels = self.labels[idx,:]


class Disfa:
    def __init__(self,config):


        self.test       = Batch(config,'test')
        self.train      = Batch(config,'train')
        self.validation = Batch(config,'validation')

        self.config = config
        self.config['image_shape'] = [self.train.images.shape[1],self.train.images.shape[2]]
        self.config['label_size'] = self.train.labels.shape[1]
