
# coding: utf-8

# In[13]:

import matplotlib.pyplot as plt
import data_array as data_array
import disfa as disfa
import numpy as np
from scipy import misc



# In[10]:

# use the first subject as example. all availabe subjects are stored in disfa.disfa_id_subj_all
id_sub = 1
# load all AU targets from subject 1
targets = disfa.disfa['AUall'][id_sub][:]


# In[11]:

# targets contain 4845 frames and 12 AUs
# see http://www.engr.du.edu/mmahoor/DISFAContent.htm for the description of the AUs, the order is ascending
targets.shape


# In[22]:

# load all images from subject 1
# images = disfa.disfa['images'][id_sub][:]
# images = scipy.misc.imresize(images,[48,48])

# load all subjects
all_images = np.empty((0, 48, 48))
#for s in [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]:
for s in [1,2,3]:
    images = disfa.disfa['images'][s][:]
    resized_images = np.zeros((images.shape[0],48,48))
    for i in xrange(images.shape[0]):
        resized_images[i,:,:] = misc.imresize(images[i,:,:],[48,48])
    del images
    all_images = np.append(all_images,resized_images,axis=0)
    print all_images.shape


exit()

# print the AU annotations and plot the image of the first sample
#print('annotated AUs are {}'.format(targets[0,:]))
#plt.imshow(images[0,:,:], cmap='gray')


# In[6]:

# as convenience function to load data from all subjects, use:
targets_all, id_array = data_array.IndicesCollection(disfa.disfa_ic_all).getitem(disfa.disfa['AUall'])


# In[ ]:

# id_array contains the subjects and frame number of each sample:
id_array[0,:]
#this means the first sample is from subject 1 frame 0


# In[7]:

a = 5000
print 'subject', id_array[a,0], 'frame', id_array[a,1]
print targets_all[a]
print targets_all.shape


# In[ ]:

import sPickle
import gzip
sPickle.s_dump(all_images, gzip.open( "disfa_images.p", "wb" ) )
sPickle.s_dump((targets_all,id_array), gzip.open( "disfa_labels.p", "wb" ) )


# In[ ]:

print 'done!'


# In[ ]:

import pickle
import gzip

#store the object
myObject = {'a':'blah','b':range(10)}
f = gzip.open('testPickleFile.pklz','wb')
pickle.dump(myObject,f)
f.close()

#restore the object
f = gzip.open('testPickleFile.pklz','rb')
myNewObject = pickle.load(f)
f.close()

print myObject
print myNewObject

