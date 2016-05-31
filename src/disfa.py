import numpy as np
import seb.data_array as data_array
import seb.disfa as disfa
from scipy import ndimage
import scipy

class Batch:
    def __init__(self,images,labels,random_batches):
        self.random_batches = random_batches
        self.images = images
        self.labels = labels
        self.counter = 0
        self.N = self.images.shape[0]

        print 'NUMBER OF SAMPELS IN BATCH:'
        print self.N

    def next_batch(self,n,overwrite_random=False):
        if n < 0:
            n = self.N

        if self.random_batches and not overwrite_random:
            idx = np.random.randint(0,self.N,size=n)
            return self.images[idx,:,:], self.labels[idx,:]
        else:
            counter = self.counter
            if counter+n > self.N:
                counter = 0
            res = self.images[counter:counter+n,:,:], self.labels[counter:counter+n,:]
            counter += 100
            self.counter = counter
            return res

class Disfa:
    def __init__(self,number_of_subjects,train_prop,valid_prop,test_prop,random_batches):
        subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]
        all_images = np.empty((0, 48,48))

        if number_of_subjects < 1:
            number_of_subjects = len(subjects)

        for s in xrange(number_of_subjects):
            images = disfa.disfa['images'][subjects[s]][:].astype(np.float32)

            resize = True
            if resize:
                # Allocate memory for downsized images
                r_images = np.zeros((images.shape[0],48,48))
                for i in xrange(images.shape[0]):
                    r_images[i,:,:] = scipy.misc.imresize(images[i,:,:],[48,48])
            else:
                r_images = images

            del images

            r_images = (r_images - np.ones(r_images.shape)*r_images.mean())/r_images.std()

            # append down_sized images to all_images
            all_images = np.append(all_images,r_images,axis=0)
            print all_images.shape

        all_labels, labels_id_array = data_array.IndicesCollection(disfa.disfa_ic_all).getitem(disfa.disfa['AUall'])

        """
        Note: I load all the labels because it was easier to write and they are very small
        """

        all_labels = all_labels.astype(float)
        all_labels = all_labels/all_labels.max()
        x,y = all_labels.shape

        weights = np.array([train_prop,valid_prop,test_prop]).astype(float)
        weights = weights/weights.sum()


        if True:
            good_samples = []
            for i in xrange(all_images.shape[0]):
                if all_labels[i,:].sum() > 0.0:
                    good_samples.append(i)

            all_images = all_images[good_samples,:,:]
            all_labels = all_labels[good_samples,:]

        self.N = all_images.shape[0]
        a = int(weights[0]*self.N)
        b = int((weights[0]+weights[1])*self.N)

        self.train = Batch(all_images[:a,:,:],all_labels[:a,:],random_batches)
        self.validation = Batch(all_images[a:b,:,:],all_labels[a:b,:],random_batches)
        self.test = Batch(all_images[b:,:,:],all_labels[b:self.N,:],random_batches)
#
#
# for i in [0,1]:
#     print test.train.next_batch(100,random=True)[i].shape
#     print test.train.next_batch(100,random=False)[i].shape
#     print test.validation.next_batch(100,random=True)[i].shape
#     print test.validation.next_batch(100,random=False)[i].shape
#     print test.test.next_batch(100,random=True)[i].shape
#     print test.test.next_batch(100,random=False)[i].shape
