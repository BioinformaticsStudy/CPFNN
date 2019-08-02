# BioinformaticsStudy
Required Package: Pytorch, Numpy.

To run the program, you should first change the train_file_path and test_file_path in the "Config" class in src/CLNN.py, then run the program by the following command:
                 python CLNN.py
                 
Note: The training data set in /data/sample_training.data is only a sample set with 10 samples each of which targets more than 450,000 methylation sites. The same training set is for demostration purpose. We don't recommend to use it for training purpose because the neural network will quickly overfit. The research uses five different GEO datasets (GSE40279, GSE73103, GSE50660, GSE42861, GSE41169) with 1843 samples; each of which targets more than 450,000 methylation sites.
