import socket
import matplotlib
if socket.gethostname() != 'ux305':
    matplotlib.use('Agg')
import matplotlib.pylab as plt

import sys,os
sys.path.append('../src')
from pyexp import PyExp

def convert(s):
    f = False
    t = True
    if s == 'contrast':
        return {'contrast':t,'face':f,'per_subject':f,'range':f}
    elif s == 'contrast_with_[-1,1]':
        return {'contrast':t,'face': f,'per_subject':f,'range':t}
    elif s == 'face':
        return {'contrast':f,'face':t,'per_subject':f,'range':f}
    elif s == 'face_ps':
        return {'contrast':f,'face':t,'per_subject':t,'range':f}
    elif s == 'none_[-1,1]':
        return {'contrast':f,'face':f,'per_subject':f,'range':t}
    elif s == 'none_[-1,1]_ps':
        return {'contrast':f,'face':f,'per_subject':t,'range':t}
    elif s == 'contrast_face':
        return {'contrast':t,'face':t,'per_subject':f,'range':f}
    elif s == 'contrast_face_ps':
        return {'contrast':t,'face':t,'per_subject':t,'range':f}
    elif s == 'none_[0,1]':
        return {'contrast':f,'face':f,'per_subject':f,'range':f}

def load_disfa(preprocessing,
               subjects='quick',
               scaling=None,
               quiet=True):
    print 'Working....'
    if quiet:
        # REDIRECT STDOUT ###################
        tmp = sys.stdout                    #
        sys.stdout = open(os.devnull,"w")   #
        #####################################
    config = PyExp(config_file='../src/config/both.yaml', path='/tmp', config_overwrite=None)

    import disfa
    reload(disfa)
    if subjects == 'quick':
        config.config['data']['train_subjects'] = [6]
        config.config['data']['validation_subjects'] = [2]
        config.config['data']['test_subjects'] = [2]
        config.config['data']['full']['resize_scale'] = 0.4
    if subjects == 'full':
        config.config['data']['train_subjects'] = [2,4,6,8,10,12,16,18,23,25,27,29,31,1,3,5,7,9,11,13,17,21,24,26,28,30,32]
        config.config['data']['validation_subjects'] = [2,4,6,8,10,12,16,18,23,25,27,29,31,1,3,5,7,9,11,13,17,21,24,26,28,30,32]
        config.config['data']['full']['resize_scale'] = 0.6

    print config.config['data']['preprocessing']
    config.config['data']['preprocessing'] = preprocessing
    print config.config['data']['preprocessing']
    data = disfa.Disfa(config['data'],debug=True,skip='train')
    if quiet:
        # UNDO STDOUT REDIRECT ##############
        sys.stdout = tmp                    #
        #####################################
    print 'Done.'
    return data

import helper
helper = reload(helper)


def show_batch_faces(batch,preprocessing,inverse=False,save=None):
    images = []
    names = []
    if inverse:
        batch.images = batch.inverse_process(batch.images)
    images.append(batch.images[0,:,:])
    names.append('A Face')
    images.append(batch.images.min(axis=0))
    names.append('Minimum Face')
    images.append(batch.images.max(axis=0))
    names.append('Maximum Face')
    images.append(batch.images.mean(axis=0))
    names.append('Mean Face')
    images.append(batch.images.std(axis=0))
    names.append('Standard Deviation\nFace')
    # if preprocessing['range']:
    #     if preprocessing['per_subject']:
    #         images.append(batch.min[0])
    #     else:
    #         images.append(batch.min)
    #     names.append('minimum '+str(0))

    # if batch.images.min() > -2.0 and batch.images.max() < 2.0:
    #     r = (batch.images.min(),batch.images.max())
    # else:
    #     r = None
    helper.plot_images(images,names,title=None,save=save)

#     names = []; images = []
#     if preprocessing['face']:
#         if preprocessing['per_subject']:
#             for i in xrange(batch.mean.shape[0]):
#                 images.append(batch.mean[i])
#                 names.append(str(i))

#     helper.plot_images(images,names,title=None,save=save)



def dataset_stats(preprocessing,
                  scaling=None,
                  subjects='quick',
                  quiet=True):

    data = load_disfa(preprocessing,
                      scaling=scaling,
                      subjects=subjects,
                      quiet=quiet)

    print data.config['preprocessing']
    if not quiet:
        print '\n\n\n'
    save = os.path.join(os.path.expanduser('~'),'Dropbox/msc_icl/541_Individual_Project/report_final/figures/')
    name = 'faces_'
    for key in preprocessing:
        if preprocessing[key]:
            name += key + '_'

    if len(name) > 0 and name[-1] == '_':
        name = name[:-1]

    save = os.path.join(save,str(name)+'.pdf')
    show_batch_faces(data.validation,preprocessing,save=save)
    # show_batch_faces(data.validation,preprocessing,save=None,inverse=True)
    del data

def main():
    methods = ['contrast','contrast_face_ps','face','face_ps','none_[-1,1]','none_[0,1]','contrast_face']
    for m in methods:
        print m
        dataset_stats(convert(m),subjects='full',quiet=False)

if __name__ == "__main__":
    main()


# def test_idx(normalisation_type=None,
#                   scaling=None,
#                   quick=True,
#                   quiet=False):
#     data = load_disfa(normalisation_type=normalisation_type,
#                       scaling=scaling,quick=quick,quiet=quiet)
#     print data.train.images.shape
#     i = []
#     N = 4
#     for j in xrange(N):
#         n = float(j)*float(data.validation.images.shape[0])/float(N)
#         i.append(int(n))
#     inp = data.validation.images[i,:,:]
#     out = data.validation.inverse_process(inp,idx=i)
# #     out = data.train.inverse_process(inp,idx=i)

#     images = []
#     names = []
#     for im in xrange(len(i)):
#         images.append(inp[im]); names.append(str(i[im]))
#         images.append(out[im]); names.append(str(i[im]))
#     plot_images(images,names,title=data.train.batch_type)

# test_idx(normalisation_type='none')
# test_idx(normalisation_type='face_ps')
# data = load_disfa(quick=True,quiet=True)
# print data.validation.subject_idx
# print data.validation.get_local_subject_id(data.validation.get_subject_id(0))
# print data.validation.get_local_subject_id(data.validation.get_subject_id(5000))
# print data.validation.get_local_subject_id(data.validation.get_subject_id(10000))
