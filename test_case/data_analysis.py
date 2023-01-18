import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv', index_col=0)

def visualize_correlation_heatmap(data, method='spearman'):
    plt.figure(figsize=(14, 8))
    correlation_matrix = data.corr(method=method)
    sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1)
    plt.title("Матрица корреляции")
    plt.show()

def create_histogram(data, column_name):
    assert column_name in data.columns, 'specified column not found in dataset.'

    plt.figure(figsize=(14, 8))
    plt.hist(data[column_name], alpha=0.5, bins=100, color='red', cumulative=True)
    plt.title(f'Cumulative histogram for {column_name} column', fontsize=16)
    plt.xlabel('Column value')
    plt.ylabel('Cumulative frequency')
    plt.grid()
    plt.show()

#visualize_correlation_heatmap(data)

#create_histogram(data, 'feature_0')