import numpy as np
import seb.data_array as data_array
import seb.disfa as disfa
from scipy import ndimage
import scipy

class Batch:
    def __init__(self,images,labels,random_batches,remove_empty_labels,type):
        self.random_batches = random_batches
        self.images = images
        self.labels = labels
        self.counter = 0
        self.N = self.images.shape[0]
        assert self.images.shape[0] == self.labels.shape[0]
        self.type = type

        if remove_empty_labels:
            good_samples = []
            for i in xrange(images.shape[0]):
                if labels[i,:].sum() > 0.0:
                    good_samples.append(i)

            self.images = images[good_samples,:,:]
            self.labels = labels[good_samples,:]

            self.N = self.images.shape[0]
            assert self.images.shape[0] == self.labels.shape[0]

        if type == 'validation':
            self.idx = np.random.randint(0,self.N,size=self.N)

        print 'NUMBER OF SAMPELS IN BATCH:'
        print self.N

    def next_batch(self,n,overwrite_random=False):
        if n < 0:
            n = self.N

        if self.random_batches and not overwrite_random:

            if self.type == 'validation':
                idx = self.idx[:n]
            else:
                idx = np.random.randint(0,self.N,size=n)
                """
                USE random.choice!
                """

            res =  self.images[idx,:,:], self.labels[idx,:]
        else:
            res = self.images[:n,:,:], self.labels[:n,:]

        return res

class Disfa:

    def image_pre_process(self,image):
        # unpack required information
        lc,rc,tc,bc = self.crop
        scale = self.resize_scale
        x,y = image.shape

        # Resize the image

        # Calculate crop indicies
        l = int(0 + x*lc)
        r = int(x - x*rc)
        u = int(0 + y*tc)
        d = int(y - y*bc)
        # Apply crop
        new_image = image[u:d,l:r]
        new_image = scipy.misc.imresize(new_image,[int(new_image.shape[0]*scale),int(new_image.shape[1]*scale)])

        return new_image

    def subject_pre_process(self,images,option='pixel'):
        if option == 'pixel':
            return (images - np.ones(images.shape)*images.mean())/images.std()
        else:
            """
            UNTESTED
            """
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

            return images


    def __init__(self,config):
        # unpack config file
        number_of_subjects = config['number_of_subjects']
        train_prop = config['train_prop']
        valid_prop = config['valid_prop']
        test_prop = config['test_prop']
        random_batches = config['random_batches']

        self.resize_scale = config['resize_scale']
        self.crop = config['crop']

        subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]

        number_of_frames_per_subject = None

        if number_of_subjects < 1:
            number_of_subjects = len(subjects)

        for s in xrange(number_of_subjects):
            if s == 0:
                # Discover dimensions of images
                images = disfa.disfa['images'][subjects[0]][:].astype(np.float32)
                number_of_frames_per_subject = images.shape[0]
                im = self.image_pre_process(images[0,:,:])
                x,y = im.shape
                all_images = np.empty((0,x,y))
                config['image_shape'] = [x,y]
            else:
                images = disfa.disfa['images'][subjects[s]][:].astype(np.float32)

            # Allocate memory for downsized images
            r_images = np.zeros((images.shape[0],x,y))
            for i in xrange(images.shape[0]):
                r_images[i,:,:] = self.image_pre_process(images[i,:,:])

            del images

            r_images = self.subject_pre_process(r_images,option='face')

            # append down_sized images to all_images
            all_images = np.append(all_images,r_images,axis=0)

        all_labels, labels_id_array = data_array.IndicesCollection(disfa.disfa_ic_all).getitem(disfa.disfa['AUall'])

        """
        Note: I load all the labels because it was easier to write and they are very small
        """

        all_labels = all_labels.astype(float)
        all_labels = all_labels/all_labels.max()
        x,y = all_labels.shape

        weights = np.array([train_prop,valid_prop,test_prop]).astype(float)
        weights = weights/weights.sum()

        aus =[6,7,8,9,10,11]
        # aus = np.arange(12)
        cropped_all_labels = np.zeros((x,len(aus)))
        config['label_size'] = len(aus)
        config['sample_number'] = x

        for i in xrange(x):
            cropped_all_labels[i,:] = all_labels[i,aus]
        all_labels = cropped_all_labels

        self.N = all_images.shape[0]

        a = float(weights[0])
        b = float(weights[0]+weights[1])

        a *= float(number_of_subjects)
        b *= float(number_of_subjects)

        a = int(round(a,0)*number_of_frames_per_subject)
        b = int(round(b,0)*number_of_frames_per_subject)

        self.train = Batch(all_images[:a,:,:],all_labels[:a,:],random_batches,config['remove_empty_labels'],'train')
        self.validation = Batch(all_images[a:b,:,:],all_labels[a:b,:],random_batches,config['remove_empty_labels'],'validation')
        self.test = Batch(all_images[b:,:,:],all_labels[b:self.N,:],random_batches,config['remove_empty_labels'],'test')

        self.config = config
