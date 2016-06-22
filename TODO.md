# TODO: FOR FRIDAY
* Literature review
* Move away from MNIST
# Complete
* READ THIS LOOKS GOOD http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial
* Ask seb about autoencoder/classifier competition



+-------------------------------------------------+
|       unlabled         |          labled        |
+------------------------+------------------------+
|  autoencoder           |  combined              |
|  autoencoder           |  classification        |
+------------------------+------------------------+

Cost functions:
* combined    A+C
* autoencoder A
* classification C


Option 1:
                alternatte:
                    A
                    A+C

Option 2:
                pretrain A
                then train C

Option 3:
                pretrain A
                then train A+C

Extend the classification layer:


/--> [500,400,300]
\--> [100,50,10]

LAST:

pre-au eval, auc,accuracy, confuion matrix per class 12x2x2 tensor2
subtract mean face or mean pixel?
higher resoltion
upper/lower face
pre-train,convert VGG16

------------

write up experiment disfa/005


For this experiment /home/luka/v/lm1015-tmp/data/2016_06_02/disfa/005 do 3 experiments:
remove intensity from training, make it binary
include intensity 1
and then both


include confusion matrices for all
test
train
validation data just once with final weights

get AU statistics for each run to make it easier to debug/understand output

plot binary graph to see how well prediction works over time in each subject video

#done
split subjects first and then load
what are the precision and recall doing at the boundaries
------------
MEETING TODO:
<!-- * For testing and validation do not exclude the all zero frames -->
* see if the binary softmax actually improved the performance
* early stopping, dump the model at the best point
-----
