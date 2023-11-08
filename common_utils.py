import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.linear_model import LinearRegression
from arch.unitroot import engle_granger
import statsmodels.api as sm
import pymc as pm
import numpy as np
import plotly.express as px
import pandas as pd
import os
import json
from scipy.linalg import eigh

def get_data(symbol, data_dir="/home/sarthak/btp/data"):
    print(symbol)
    path = os.path.join(data_dir, symbol + ".json")
    try:
        with open(path) as f:
            data = json.load(f)
        if data["t"] == None:
            return None
        df = pd.DataFrame({
            'datetime': data['t'],
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index, unit='s')
        return df
    except Exception as e:
        print(e)
        return None


def train_test_split(df, train_size=0.8):
    train_size = int(len(df) * train_size)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    return train, test

def get_common_data(df_arr):
    common_dates = df_arr[0].index
    for df in df_arr:
        common_dates = common_dates.intersection(df.index)
    common_dates = common_dates.sort_values()
    common_df_arr = []
    for df in df_arr:
        common_df_arr.append(df.loc[common_dates])
    return common_df_arr

# def get_stocks_common_interval(data):


def find_cointegrated_pairs(data, threshold=0.05):
    symbols = list(data.keys())
    pairs = []
    n = len(symbols)
    pvalues = np.ones((n, n))
    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            print(symbols[i], symbols[j])
            S1 = data[symbols[i]]
            S2 = data[symbols[j]]
            S1 = S1['close']
            S2 = S2['close']
            s = get_common_data([S1, S2])
            if len(s[0]) == 0:
                continue
            try:
                result = coint(s[0], s[1])
            except ZeroDivisionError:
                continue
            except ValueError:
                continue
            pvalue = result[1]
            pvalues[i, j] = pvalue
            score = result[0]
            score_matrix[i, j] = score
            if pvalue < threshold:
                pairs.append((symbols[i], symbols[j]))
    return score_matrix, pvalues, pairs

def normalize(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def get_all_symbols(data_dir = "/home/sarthak/btp/data"):
    files = os.listdir(data_dir)
    symbols = []
    for file in files:
        if file.endswith(".json"):
            symbols.append(file[:-5])
    return symbols


def get_all_data(data_dir = "/home/sarthak/btp/data"):
    symbols = get_all_symbols(data_dir)
    data = {}
    for symbol in symbols:
        x = get_data(symbol, data_dir)
        if x is not None and len(x) > 1000:
            data[symbol] = x
    return data

def filter_stocks(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    tickers = list(data.keys())
    for ticker in tickers:
        if data[ticker].index[0] > start_date or data[ticker].index[-1] < end_date:
            del data[ticker]
    return data

def pair_hedge(df1, df2):
    df1 = df1['close']
    df2 = df2['close']
    s = get_common_data([df1, df2])
    scaled_df1 = s[0].apply(scale)
    scaled_df2 = s[1].apply(scale)
    model_randomwalk = pm.Model()
    with model_randomwalk:
        sigma_alpha = pm.Exponential('sigma_alpha', 50.)    
        alpha = pm.GaussianRandomWalk('alpha', 
                                    sd=sigma_alpha, 
                                    shape=len(scaled_df1))
        
        sigma_beta = pm.Exponential('sigma_beta', 50.) 
        beta = pm.GaussianRandomWalk('beta', 
                                    sd=sigma_beta, 
                                    shape=len(scaled_df1))
    
        regression = alpha + beta * scaled_df1
        sd = pm.HalfNormal('sd', sd=.1)
        likelihood = pm.Normal('y', 
                           mu=regression, 
                           sd=sd, 
                           observed=scaled_df2)
        
        trace_rw = pm.sample(tune=2000, 
                         draws=200,
                         cores=1,
                         target_accept=.9)
        
        return likelihood, trace_rw


def pca(data, components=3):
    tickers = list(data.columns)
    dfs = []
    for ticker in tickers:
        dfs.append(data[ticker])

    min_sz = len(data[tickers[0]])

    for ticker in tickers:
        if len(data[ticker]) < min_sz:
            min_sz = len(data[ticker])

  
    returns_list = []
    for ticker in tickers:
        r = data[ticker].pct_change().dropna().values * 100  # using pct_change
        r = r[-(min_sz-1):]  # Adjusted the indexing to match the MATLAB code
        returns_list.append(r)

    returns_matrix = np.column_stack(returns_list)
    date_diff = returns_matrix.shape[0] - 1

    returns_matrix = returns_matrix[:date_diff, :]

    # Perform PCA
    A = returns_matrix
    mu = np.mean(A, axis=0)
    stdevs = np.std(A, axis=0)
    T = (A - mu) / stdevs

    pca = PCA(n_components = components)
    ## Apply PCA to the data
    principalComponents = pca.fit_transform(T.transpose())
    print(pca.explained_variance_)
    ## Add the pca results to a data frame
    columns = [f'pc{i}' for i in range(1, components+1)]
               
    principalDf = pd.DataFrame(data = principalComponents
                ,columns = columns)
    ## Get the stock names from the data set that removed stocks with missing values
    principalDf.index = [list(data.columns)]

   

    print(principalDf)
    return principalDf

def optics(principalDf):
     # Apply OPTICS
    optics_clustering = OPTICS(min_samples = 3).fit(principalDf)
    ## make a variable containing the labels for each stock
    labels = optics_clustering.labels_
    ## number of clusters created
    number_of_clusters = np.unique(labels)
    ## Add labels array to the dataframe containing the principle components for the stocks
    data_clusters = principalDf
    ## The feature cluster containing the labels for each stocks is added
    data_clusters['cluster'] = labels
        
    # Plot the 3D clusters
    ## Plot the clusters created with OPTICS clustering
    # fig_clusters = px.scatter_3d(data_clusters, x='pc1', y='pc2', z='pc3',
    #             color='cluster')
    # fig_clusters.show()
    print(number_of_clusters)
    print(data_clusters)
    return data_clusters, number_of_clusters

# Function to choose which cluster you want to see the mean reverting pairs from

def cluster_pairs(select_cluster, data_clusters, data):
    
    # Sort dataframes seperate for each cluster
    cluster_data_indexes  = data_clusters.loc[data_clusters['cluster'] == select_cluster]
    cl = cluster_data_indexes.reset_index()
    cl = cl['level_0']
    cluster_data_indexes = cl
    
    # Apply all 4 pairs selection criteria 

    number_stocks_in_cluster = len(cluster_data_indexes)
    combinations = np.int((number_stocks_in_cluster*(number_stocks_in_cluster-1))/2)

    # Function to calculate the mean revertion in the spread of the stock pairs
    def get_hurst_exponent(time_series, max_lag=20):

        lags = range(2, max_lag)

        # variances of the lagged differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)

        return reg[0]

    # Empty data frame to fill with the pairs selection criteria
    data_selection_crit = pd.DataFrame(index=range(combinations),columns=range(6))
    data_selection_crit.columns =['pair1', 'pair2', 'Cointegration', 'Mean revertion', 'Half-life','Cross']

    pos = 0
    for i in range(0,(number_stocks_in_cluster)):
        for j in range(i+1,number_stocks_in_cluster):
            # Fill in the pair
            data_selection_crit.iloc[pos,0] = cluster_data_indexes[i]
            data_selection_crit.iloc[pos,1] = cluster_data_indexes[j]
            # Engler-granger to determine cointegration
            engel_granger_test = engle_granger(data[cluster_data_indexes[i]],data[cluster_data_indexes[j]])
            pval = engel_granger_test.pvalue

            data_selection_crit.iloc[pos,2] = pval

            # Hurst exponent
            ## Apply OLS regression to get the hedge ratio 
            model = LinearRegression().fit(np.array(data[cluster_data_indexes[i]]).reshape((-1, 1))
                    , np.array(data[cluster_data_indexes[j]]))
            hedge_ratio = float(model.coef_)
            ## Calculate the spread using the hedge ratio
            spread = np.log(data[cluster_data_indexes[j]]) - np.multiply(np.log(data[cluster_data_indexes[i]]),hedge_ratio)
            for lag in [20]:
                hurst_exp = get_hurst_exponent(spread.values, lag)

            data_selection_crit.iloc[pos,3] = hurst_exp

            # Half-life of the spread determined in days
            z_lag = np.roll(spread,1)
            z_lag[0] = 0
            z_ret = spread - z_lag
            z_ret[0] = 0

            # Use a function to determine the half-life in days 
            z_lag2 = sm.add_constant(z_lag)

            model = sm.OLS(z_ret,z_lag2)
            res = model.fit()

            halflife = -np.log(2) / res.params[1] #in days

            data_selection_crit.iloc[pos,4] = halflife

            #How many of the month in the trade period did the spread cross the mean at least once

            s = spread.to_frame(name='spread')
            s.loc[s['spread'] > np.mean(s['spread']), 'count'] = 1
            s.loc[s['spread'] < np.mean(s['spread']), 'count'] = 0
            s.reset_index(inplace=True)
            s = s.rename(columns = {'index':'Date'})
            s['switch'] = ''
            s.iloc[0,3] = 0
            for k in range(1,len(s)):
                if s.iloc[k,2] + s.iloc[k-1,2] == 0 or s.iloc[k,2] + s.iloc[k-1,2] == 2:
                    s.iloc[k,3] = 0
                else:
                    s.iloc[k,3] = 1

            s['Date'] = s['Date'].dt.strftime('%m-%Y')

            pivot = s.pivot_table(values='switch', index = 'Date', \
                                 aggfunc= 'sum')

            piv = pivot.reset_index()
            count_zeros = piv[piv['switch']==0].count()['switch']
            months_crossed = len(piv) - count_zeros

            data_selection_crit.iloc[pos,5] = months_crossed

            pos = pos + 1

    return data_selection_crit
    