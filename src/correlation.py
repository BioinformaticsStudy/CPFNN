import numpy as np
from scipy import stats

def calculate_correlation(data):
    spearman_result = []
    pearson_result = []
    age = data[:,0]
    for i in range(1,len(data[0])):
        s_rho,pvalue = stats.spearmanr(data[:,i],age)
        p_rho,pvalue = stats.pearsonr(data[:,i],age)
        spearman_result.append(s_rho)
        pearson_result.append(p_rho)
    spearman_result = np.array(spearman_result)
    pearson_result = np.array(pearson_result)
    return spearman_result,pearson_result
