# TODO:
* Big plan for thursday
    * Create mangled MNIST data set
    * Figure out if using it improves the classification
    * Experiments:


* Literature review
* Data reporting system. I think pickle will be good so that I can re print the data nicely.
# Complete
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
