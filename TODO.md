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
