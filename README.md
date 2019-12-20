# Parser

To use the GEO_parser.py, you need to provide a txt file named geo_info.txt which records information about GEO file you want to use. You also need to download the cleaned_sites_common.txt file to help parse to filter the unrelated columns. Each line is a GEO dataset. The format of geo_info.txt is following:

GEO_Dataset_ID, identified_age_string, identified_disease_string, identified_healthy_string ,tissue_type_label, tissue_type_name

0 is for missing column.

One specific example is provied below and more specific examples can be found on geo_info.txt.

GSE67530,characteristics_ch1.1.age,characteristics_ch1.3.ards,NA,0,0



# CPFNN
Required Package: Pytorch, Numpy.

To run the program, you should first change the train_file_path and test_file_path in the "Config" class in src/CFPNN.py, then run the program by the following command:

                 python CFPNN.py .
The program will predict the biological age for each testing people. 

Note: The training data in /data/sample_training.csv is only sample data with 10 samples each of which targets more than 450,000 methylation sites. The sameple training set is for demostration purpose. We don't recommend reader to use it for training purpose because the neural network will quickly overfit. The research uses five different GEO datasets (GSE40279, GSE73103, GSE50660, GSE42861, GSE41169) with 1843 samples; each of which targets more than 450,000 methylation sites.
