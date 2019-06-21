import numpy as np 
from numpy.linalg import norm
import scipy.stats
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def cal_vector(op, u, v):
    """Calculate some operations of 2 vectors u and v"""
    result = None
    if op == "+":
        result = u + v 
    elif op == "-":
        result = u - v
    elif op == "*":
        result = u * v
    elif op == "/":
        result = u / v
    else:
        result = None
    return result

def cal_vector_norm(t, v):
    """Calculate vector norm, type of norm t"""
    result = None
    if t == "L1":
        result = norm(v, 1)
    elif t == "L2":
        result = norm(v, 2)
    elif t == "Max":
        result = norm(v, math.inf)
    else:
        result = None
    return result

def cal_matrices(op, m1, m2):
    """Calculate some operations of 2 matrices m1 and m2"""
    result = None
    if op == "+":
        result = m1 + m2 
    elif op == "-":
        result = m1 - m2
    elif op == "*":
        result = m1 * m2
    elif op == "/":
        result = m1 / m2
    else:
        result = None
    return result

def create_matrix(m, n):
    """Create matrix based on size mxn"""
    lst = []
    for i in range(m):
        lst_sub = []
        for j in range(n):
            s = "M[" + str(i + 1) + "," + str(j + 1) + "]:"
            x = eval(input(s))
            lst_sub.append(x)
        lst.append(lst_sub)
    return np.array(lst)

def create_vector(n):
    """Create vector based on size n"""
    lst = []
    for i in range(n):
        s = "v[" + str(i + 1) + "]:"
        x = eval(input(s))
        lst.append(x)
    return np.array(lst)

def create_matrix_random(m, n, start, end):
    """Create matrix based on size mxn, random value from start to end params"""
    lst = []
    for _ in range(m):
        lst_sub = []
        for _ in range(n):
            x = random.randint(start, end + 1)
            lst_sub.append(x)
        lst.append(lst_sub)
    return np.array(lst)

def create_vector_random(n, start, end):
    """Create vector based on size m, random value from start to end params"""
    lst = []
    for _ in range(n):
        x = random.randint(start, end + 1)
        lst.append(x)
    return np.array(lst)

def create_matrix_positive_definite(m, n, start, end):
    E = None
    flag = False
    while flag == False:
        E =  create_matrix_random(m, n, start, end)
        for i in range(E.shape[0]):
            for j in range(i):
                E[j][i] = E[i][j]
        test = np.linalg.eigvalsh(E)
        flag = np.all(test>0)
    return E

def pca(dataframe, number_principal_components):
    pca = PCA(number_principal_components)
    pca.fit(dataframe)
    transform = pca.transform(dataframe)
    dict_pca_info = {"components": pca.components_, "explained_variance": pca.explained_variance_,
                    "explained_variance_ratio": pca.explained_variance_ratio_}
    return pca, transform, dict_pca_info

def mean_confidence_interval(data, confidence=0.95):
    """Return confidence interval of input data, default confidence=0.95"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h, m - h, m + h

def frequency_plot(array_data, figsize=(8,4)):
    """Plot histogram and distribution of input data with customized figsize"""
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.hist(array_data)
    plt.subplot(1, 2, 2)
    sns.distplot(array_data)
    plt.show()

def scatter_plot(x_data, y_data, xlabel, ylabel, marker=".", linestype="none", figsize=(8,6)):
    plt.figure(figsize=figsize)
    plt.plot(x_data, y_data,
                marker=marker, linestyle=linestype)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def box_plot(dataframe, col_name1, col_name2, figsize=(8,6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    sns.boxplot(dataframe[col_name1])
    plt.subplot(1, 2, 2)
    sns.boxplot(dataframe[col_name2])
    plt.show()

def joint_plot(dataframe, x_col_name, y_col_name, figsize=(8,6)):
    plt.figure(figsize=figsize)
    sns.jointplot(x=x_col_name, y=y_col_name, data=dataframe)
    plt.show()

def distribution_plot(array_data, xlabel, ylabel, extensions=False, bins=100, kde=True, color="blue",
                        hist_kws={"linewidth": 15,'alpha':1}, figsize=(8,6)):
    if extensions:
        ax = sns.distplot(array_data,
                        bins=bins,
                        kde=kde,
                        color=color,
                        hist_kws=hist_kws)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        plt.show()
    else:
        plt.figure(figsize=figsize)
        sns.distplot(array_data)
        plt.show()

def generate_random_variable_sample(size, loc, scale):
    """Generate random sample based on predefied size, loc (mean), scale (std)"""
    return scipy.stats.norm.rvs(size=size, loc=loc, scale=scale)

def common_statistic_array(array_data, percentiles):
    """Compute common statistics figures (mean, median, mode, std, var, etc)"""
    assert type(array_data).__name__ == "ndarray"
    assert type(percentiles).__name__ == "ndarray"
    mean_ = array_data.mean()
    max_ = array_data.max()
    min_ = array_data.min()
    mode_ = scipy.stats.mode(array_data)
    median_ = np.median(array_data)
    range_ = np.ptp(array_data)
    percentiles_ = np.percentile(array_data, percentiles)
    iqr_ = scipy.stats.iqr(array_data)
    var_ = np.var(array_data)
    std_ = np.std(array_data)
    skew_ = scipy.stats.skew(array_data)
    kur_ = scipy.stats.kurtosis(array_data) # default Fisher=True, kur - 3
    kur_pearson_ = scipy.stats.kurtosis(array_data, fisher=False)
    z_score_ = scipy.stats.zscore(array_data)
    outliers_ = z_score_[[(n <=-2.5)|(n>=2.5) for n in z_score_]]
    indexes_outliers_ = [z_score_.tolist().index(i) for i in outliers_]
    sigma = var_ ** 0.5
    three_sigmas_ = [sigma, sigma * 2, sigma * 3]
    three_sigmas_std_ = [std_, std_ * 2, std_ * 3]
    dict_results = {"mean": mean_, "max": max_, "min": min_, "mode": mode_, "median": median_, "range": range_,
                    "percentile": percentiles_, "IQR": iqr_, "var": var_, "std": std_,
                    "skew": skew_, "kurtosis": kur_, "kurtosis_pearson": kur_pearson_, "z_score": z_score_, 
                    "outlier_values": array_data[indexes_outliers_], "index_outliers": indexes_outliers_,
                    "three_sigma_var": three_sigmas_, "three_sigma_std": three_sigmas_std_}
    return dict_results

def common_statistic_2_arrays(array_data_1, array_data_2):
    assert type(array_data_1).__name__ == "Series"
    assert type(array_data_2).__name__ == "Series"
    pearson_correlation = array_data_1.corr(array_data_2)
    spearman_correlation = array_data_1.corr(array_data_2, method="spearman")
    return pearson_correlation, spearman_correlation

def create_array_from_txt(file_name):
    """From (int) txt file to numpy array"""
    f = open(file_name, "r")
    content = f.read()
    data = content.split()
    data = list(map(int, data))
    array_data = np.array(data)
    f.close()
    return array_data

def frequency_table(array_data, min_range, max_range, interval, plot=False):
    lst_i = []
    lst_j = []
    for i in range(min_range, max_range, interval):
        j = i + interval - 1
        lst_i.append(i)
        lst_j.append(j)
    pairs = [str(i) + "-" + str(j) for i, j in zip(lst_i, lst_j)]
    freq = pd.Series()
    for pair, i, j in zip(pairs, lst_i, lst_j):
        freq[pair] = np.extract((array_data >= i) & (array_data <= j), array_data).size
    if plot:
        plt.bar(freq.index, freq)
        plt.show()
    return freq
    
def calculate_z_score(value, array_data):
    return (value - np.mean(array_data)) / np.std(array_data)

def print_index_k_fold(x_value, y_value, test_size, n_split, random_state):
    x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=test_size)
    cv = KFold(n_splits=n_split, random_state=random_state)
    time = 0
    for train_index, test_index in cv.split(x_value):
        time += 1
        print("*** Time:", time)
        print("- Train Index: ", train_index.tolist())
        print("- Test Index: ", test_index.tolist())
        print("\n")
        x_train, x_test = x_value.iloc[train_index.tolist()], x_value.iloc[test_index.tolist()]
        y_train, y_test = y_value.iloc[train_index.tolist()], y_value.iloc[test_index.tolist()]
    return x_train, x_test, y_train, y_test

def t_test(array_data_1, array_data_2, alpha=0.5):
    t, p = scipy.stats.ttest_ind(array_data_1, array_data_2)
    print("t=%.2f, p=%.2f" % (t,p))
    if p > alpha:
        print("Accept null hypothesis")
    else:
        print("Reject null hypothesis")

def chi_square_test(table, prob=0.95):
    stat, p, dof, expected = scipy.stats.chi2_contingency(table)
    critical = scipy.stats.chi2.ppf(prob, dof)
    print("probability=%.3f, critical=%.3f, stat=%.3f" % (prob, critical, stat))
    if abs(stat) >= critical:
        print("Reject Ho")
    else:
        print("Failed to reject Ho")
    dict_results = {"stat": stat, "p": p, "dof": dof, "critical": critical, "prob": prob, "expected": expected}
    return dict_results

