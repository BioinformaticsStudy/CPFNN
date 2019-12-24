####################################################################################
# Description:
# v0.9 version
# merge implemented
# 
# Documentation of GEOparse:
# https://buildmedia.readthedocs.org/media/pdf/geoparse/latest/geoparse.pdf
# 
# cpg cite value's decimal digits kept: 4
#
# Usage of this script:
# pip install GEOparse
# python geo_parser.py
# 
# Output: 
# output.csv
# columns look like:
# age,availability,cg00374717,cg00436603,...
#
# todo:
# > use argparse to add parameter parser
#       python geo_parser.py -m metadata.csv
# > design metadata.csv
#       columns:
#       geo_id,age_string,disease_string,file_location
#       ex:   GSE41169,characteristics_ch1.6.age,characteristics_ch1.7.diseasestatus (1=control,..),../Data/GSE41169.csv
####################################################################################

import GEOparse
import pandas as pd
import numpy as np

# global variables:
common_cpg_sites_path = './cleaned_sites_common.txt'
availability_col_name = 'availability'
age_col_name = 'age'
# Set the log level.
GEOparse.logger.set_verbosity("ERROR")

def divider():
    print("-"*25)

def get_cpg_ref_n_value_columns_name(gse):
    first_sample_id = next(iter(gse.gsms))
    tmp_sample_table = gse.gsms[first_sample_id].table
    cpg_id_ref_column_name = tmp_sample_table.columns[0]  # first column is the cpg_id_ref_column
    print()
    print("cpg_id_ref_column_name in [", gse.name,"]: ", cpg_id_ref_column_name)
    cpg_value_column_name = tmp_sample_table.columns[1]   # second column is the cpg_value_column
    print("cpg_value_column_name in [", gse.name,"]: ", cpg_value_column_name)
    len_of_cpgsites = len(tmp_sample_table)
    print("number_of_cpgsites in [", gse.name,"]: ", len_of_cpgsites)
    return cpg_id_ref_column_name, cpg_value_column_name

# return a sub aggregated dataset
# geo_id = 'GSE65638' # sample size: 16
# age_str = 'characteristics_ch1.1.age'  # GSE65638's
# disease_str = '0'
# disease_keyword = '0'
# measure_str = '0' 
# measure_keyword = '0'

def parse_every_dataset(geo_id, age_str, disease_str, disease_keyword, measure_str, measure_keyword, common_cpg_sites):
    divider()
    print("Start parsing dataset: [", geo_id, "]")

    gse = GEOparse.get_GEO(geo=geo_id, destdir="./", silent=True)

    cpg_id_ref_column_name, cpg_value_column_name = get_cpg_ref_n_value_columns_name(gse)
    sub_aggregated_dataset = pd.DataFrame()

    sample_id_col = pd.Series()

    for sample_id in gse.gsms:
        sample_id_col = sample_id_col.append(pd.Series([sample_id]))
        sample_df = gse.gsms[sample_id].table
        # round every cpg value to 4 decimal points
        cpg_sites = sample_df[cpg_value_column_name].round(4).astype(np.float32)    # float32 4 digits
        # print(cpg_sites.head(10))
        sub_aggregated_dataset = pd.concat([sub_aggregated_dataset, cpg_sites], axis=1, ignore_index=True)
        # print("Type of columns: \n", sub_aggregated_dataset.dtypes)

    sub_aggregated_dataset = sub_aggregated_dataset.T
    # change the columns to cpg sites
    sub_aggregated_dataset.columns = sample_df[cpg_id_ref_column_name]

    # set the index to be sample_id   in order to add age/disease by sample_id
    sub_aggregated_dataset.index = sample_id_col

    # obtain age dataframe of this dataset
    age_df = gse.phenotype_data[[age_str]]
    # then merge dataframes by their index, which is the sample_id
    sub_aggregated_dataset = pd.merge(sub_aggregated_dataset, age_df, left_index=True, right_index=True)
    # change columns' name:
    sub_aggregated_dataset.rename(columns={age_str:age_col_name}, inplace=True)

    # add availability(geo_id) to each row of the df
    sub_aggregated_dataset[availability_col_name] = gse.name

    if disease_str is not '0':
        patient_df = gse.phenotype_data[[disease_str]]
        sub_aggregated_dataset = pd.merge(sub_aggregated_dataset, patient_df, left_index=True, right_index=True)
        sub_aggregated_dataset = sub_aggregated_dataset.loc[sub_aggregated_dataset[disease_str]==disease_keyword]
        sub_aggregated_dataset = sub_aggregated_dataset.drop([disease_str], axis=1)    

    if measure_str is not '0':
        measure_df = gse.phenotype_data[[measure_str]]
        sub_aggregated_dataset = pd.merge(sub_aggregated_dataset, measure_df, left_index=True, right_index=True)
        sub_aggregated_dataset = sub_aggregated_dataset.loc[sub_aggregated_dataset[measure_str]==measure_keyword]
        sub_aggregated_dataset = sub_aggregated_dataset.drop([measure_str], axis=1)

    # sort by columns
    sub_aggregated_dataset.sort_index(axis=1, inplace=True)
    # filter columns by common cpg sites
    sub_aggregated_dataset = sub_aggregated_dataset.loc[:, sub_aggregated_dataset.columns.isin(common_cpg_sites)]

    print("shape of sub: ", sub_aggregated_dataset.shape)
    # print("sort and filter are good!")
    print("Column(index) number of sub aggregated dataset of [", geo_id, "]: ", sub_aggregated_dataset.index.size)
    print("Finished parsing dataset: [", geo_id, "]")
    divider()

    return sub_aggregated_dataset

def get_common_cpg_sites_set(filepath):
    with open(filepath, 'r') as file:
    	common_cpg_sites = file.readlines()
    common_cpg_sites_combined_with_age_ava_cols = [x.strip() for x in common_cpg_sites] + [age_col_name, availability_col_name]
    return sorted(common_cpg_sites_combined_with_age_ava_cols)

def get_geo_dict(filepath):
    geo_dict = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            attributes = line.split(",")
            value_lst = [attributes[1].strip(), attributes[2].strip(),attributes[3].strip(),attributes[4].strip()]
            geo_dict[attributes[0]] = value_lst
    return geo_dict

def main():
    # read common cpg sites and saved as a set
    common_cpg_sites = get_common_cpg_sites_set(filepath=common_cpg_sites_path)

    aggregated_dataset = pd.DataFrame()

    # main loop
    #geo_dict = geo_dict_few
    geo_dict = get_geo_dict("./geo_info.txt")
    print("Going to parse: ", list(geo_dict.keys()), "\n")

    #parse each dataset and contact them
    for geo_id in geo_dict.keys():
        value_list = geo_dict[geo_id]
        age_str = value_list[0]
        disease_str = value_list[1]
        disease_keyword = value_list[2]
        measure_str = value_list[3]
        measure_keyword = value_list[4]

        try:
        	sub_aggregated_dataset = parse_every_dataset(geo_id, age_str, disease_str, disease_keyword, measure_str, measure_keyword, common_cpg_sites)
        	aggregated_dataset = pd.concat([aggregated_dataset, sub_aggregated_dataset], ignore_index=True)
        except Exception as e:
        	print('Error occured while parsing dataset[', geo_id,']: ', str(e))

    # write to output file without index keeps auto decimal points and header(column name)
    aggregated_dataset.to_csv('./merged_dataset.csv', index=False, header=True, float_format='%g')
    print()
    print('Finish aggregated all datasets')

    print("Final shape of merged dataset: ", aggregated_dataset.shape)

if __name__ == '__main__':
	main()
