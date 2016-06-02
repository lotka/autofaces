import tensorflow as tf
import numpy as np
import sys
from scipy.stats import hmean
from sklearn.metrics import roc_curve, auc

def safe_div(x,y):
    if y == 0.0 and x == 0.0:
        return 0.0
    else:
        return x/y

def metric(autoencoder,sess,y,mnist,x,log_file=None,no_classes=10):
    pred = autoencoder['output_class']
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    y_p = tf.argmax(pred, 1)

    test_arrays, test_label = mnist.test.next_batch(100)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_arrays, y:test_label})

    y_true = np.argmax(test_label,1)

    confuse =  np.zeros((no_classes,no_classes),dtype=int)

    if log_file != None:
        log = open(log_file, "w")
        sys.stdout = log
        sys.stderr = log

    N = len(y_true)
    for i in xrange(N):
        confuse[y_true[i],y_pred[i]] += 1

    print 'class precision recall'
    tot_prec = 0
    tot_recall = 0
    for i in xrange(no_classes):
        prec = safe_div(float(confuse[i,i]),float(confuse[:,i].sum()))
        recall = safe_div(float(confuse[i,i]),float(confuse[i,:].sum()))
        print i,'\t', round(prec,2),'\t', round(recall,2)
        tot_recall += recall
        tot_prec += prec

    print 'avg\t',round(tot_recall/float(no_classes),2),'\t', round(tot_prec/float(no_classes),2)
    print confuse

    if log_file != None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    log.close()


    return

def confuse_2d(y_true,y_pred):
    no_classes = 2
    confuse =  np.zeros((no_classes,no_classes),dtype=int)

    N = len(y_true)
    for i in xrange(N):
        confuse[y_true[i],y_pred[i]] += 1

    prec = safe_div(float(confuse[1,1]),float(confuse[:,1].sum()))
    recall = safe_div(float(confuse[1,1]),float(confuse[1,:].sum()))
    if prec == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = hmean([prec,recall])
    return prec,recall,f1,confuse

def multi_eval(y_pred,y_true,threshold=0.6):

    yp_shape = y_pred.shape
    yt_shape = y_true.shape
    print yp_shape
    print yt_shape
    assert yp_shape == yt_shape
    runs, classes = yt_shape
    y_pred_binary = y_pred.copy()

    for r in xrange(runs):
        for c in xrange(classes):
            # y_pred.mean()+y_pred.std()
            if y_pred_binary[r,c] > threshold:
                y_pred_binary[r,c] = 1.0
            else:
                y_pred_binary[r,c] = 0.0

            if y_true[r,c] > threshold:
                y_true[r,c] = 1.0
            else:
                y_true[r,c] = 0.0

    y_pred = y_pred.astype(float)
    y_true = y_true.astype(int)
    y_pred_binary = y_pred_binary.astype(int)


    res = np.zeros((classes,4))
    confusion_matrices = []
    for i in xrange(classes):
        prec,recall,f1,confuse = confuse_2d(y_true[:,i],y_pred_binary[:,i])
        a = y_true[:,i].sum() != 0
        b = y_true[:,i].sum() != len(y_true[:,i])
        c = y_pred[:,i].sum() != 0
        d = y_pred[:,i].sum() != len(y_pred[:,i])
        if a and b and c and d:
            # print 'SUMS', y_true[:,i].sum(), y_pred[:,i].sum()
            fpr, tpr, thresholds = roc_curve(y_true[:,i],y_pred[:,i])
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0
        res[i,0] = prec
        res[i,1] = recall
        res[i,2] = f1
        res[i,3] = roc_auc
        confusion_matrices.append(confuse)

    return res,confusion_matrices
