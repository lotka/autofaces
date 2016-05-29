import numpy as np
import seb.data_array as data_array
import seb.disfa as disfa
from scipy import ndimage
import scipy

class Batch:
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.N = labels.shape[0]
        self.counter = 0

    def next_batch(self,n,random):
        if n < 0:
            n = self.N

        if random:
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
    def __init__(self,number_of_subjects,train_prop,valid_prop,test_prop):
        subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]
        all_images = np.empty((0, 48, 48))
        #for s in [1]:

        if number_of_subjects < 1:
            number_of_subjects = len(subjects)

        for s in xrange(number_of_subjects):
            images = disfa.disfa['images'][subjects[s]][:]
            resized_images = np.zeros((images.shape[0],48,48))
            for i in xrange(images.shape[0]):
                resized_images[i,:,:] = scipy.misc.imresize(images[i,:,:],[48,48]).astype(float)/float(255)
            del images
            all_images = np.append(all_images,resized_images,axis=0)
            print all_images.shape


        all_labels, labels_id_array = data_array.IndicesCollection(disfa.disfa_ic_all).getitem(disfa.disfa['AUall'])

        """
        Note: I load all the labels because it was easier to write and they are very small
        """

        all_labels = all_labels.astype(float)
        x,y = all_labels.shape
        for i in xrange(x):
            for j in xrange(y):
                if all_labels[i,j] > 2:
                    all_labels[i,j] = 1.0
                else:
                    all_labels[i,j] = 0.0

        weights = np.array([train_prop,valid_prop,test_prop]).astype(float)
        weights = weights/weights.sum()

        self.N = all_images.shape[0]
        a = int(weights[0]*self.N)
        b = int((weights[0]+weights[1])*self.N)
        print a,b

        self.train = Batch(all_images[:a,:,:],all_labels[:a,:])
        self.validation = Batch(all_images[a:b,:,:],all_labels[a:b,:])
        self.test = Batch(all_images[b:,:,:],all_labels[b:self.N,:])
#
#
# for i in [0,1]:
#     print test.train.next_batch(100,random=True)[i].shape
#     print test.train.next_batch(100,random=False)[i].shape
#     print test.validation.next_batch(100,random=True)[i].shape
#     print test.validation.next_batch(100,random=False)[i].shape
#     print test.test.next_batch(100,random=True)[i].shape
#     print test.test.next_batch(100,random=False)[i].shape
