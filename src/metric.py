import tensorflow as tf
import numpy as np
from sklearn import metrics

def safe_div(x,y):
    if y == 0.0 and x == 0.0:
        return 0.0
    else:
        return x/y

def metric(autoencoder,sess,y,mnist,x):
    pred = autoencoder['output_class']
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    y_p = tf.argmax(pred, 1)

    test_arrays, test_label = mnist.test.next_batch(100)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_arrays, y:test_label})

    y_true = np.argmax(test_label,1)

    confuse =  np.zeros((10,10),dtype=int)

    N = len(y_true)
    for i in xrange(N):
        confuse[y_true[i],y_pred[i]] += 1

    print 'class precision recall'
    tot_prec = 0
    tot_recall = 0
    for i in xrange(10):
        prec = safe_div(float(confuse[i,i]),float(confuse[:,i].sum()))
        recall = safe_div(float(confuse[i,i]),float(confuse[i,:].sum()))
        print i,'\t', round(prec,2),'\t', round(recall,2)
        tot_recall += recall
        tot_prec += prec

    print 'avg\t',round(tot_recall/float(10),2),'\t', round(tot_prec/float(10),2)
    print confuse

    return
