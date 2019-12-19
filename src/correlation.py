import numpy as np
from scipy import stats

#calculate pearson and spearman correlation between individual CpG sites and age
def calculate_correlation(data):
    spearman_result = [] #list that store return spearman correlation result
    pearson_result = [] #list that store return pearson correlation result
    age = data[:,0] #select age column from imput data
    for i in range(1,len(data[0])): #loop for all CpG sites
        s_rho,pvalue = stats.spearmanr(data[:,i],age) #calculate spearman correlation and p value between methylation value and age
        p_rho,pvalue = stats.pearsonr(data[:,i],age) #calculate pearson correlation and p value between methylation value and age
        spearman_result.append(s_rho) #add spearman correlation to the return list
        pearson_result.append(p_rho) #add pearson correlation to the return list
    spearman_result = np.array(spearman_result) #convert spearman correlation return list to numpy array
    pearson_result = np.array(pearson_result) #convert pearson correlation return list to numpy array
    return spearman_result,pearson_result #return pearson correlation and spearman correlation
