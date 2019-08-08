# BioinformaticsStudy
Required Package: Pytorch, Numpy.

To run the program, you should first change the train_file_path and test_file_path in the "Config" class in src/CFPNN.py, then run the program by the following command:

                 python CFPNN.py .
The program will predict the biological age for each testing people. 

Note: The training data in /data/sample_training.csv is only sample data with 10 samples each of which targets more than 450,000 methylation sites. The sameple training set is for demostration purpose. We don't recommend reader to use it for training purpose because the neural network will quickly overfit. The research uses five different GEO datasets (GSE40279, GSE73103, GSE50660, GSE42861, GSE41169) with 1843 samples; each of which targets more than 450,000 methylation sites.
