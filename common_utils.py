import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import json

def get_data(symbol, data_dir="/home/sarthak/btp/data"):
    print(symbol)
    path = os.path.join(data_dir, symbol + ".json")
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
        if x is not None:
            data[symbol] = x
    return data


