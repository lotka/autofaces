# TODO: FOR FRIDAY
* Literature review
* Use that paper to add convolutions
* ROC shit
* Move away from MNIST
* Data reporting system.
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
higher resoltion
upper/lower face
pre-train,convert VGG16



------------


lmsq
cross_entropy
prec
recall
f1
auc
confusion matrix 2x2
