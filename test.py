from common_utils import *
import pandas as pd
data = pd.read_csv('./data/cleaned_stocks.csv')
data.set_index('Date', inplace=True)
data.index = pd.to_datetime(data.index)
# data = filter_stocks(data, '2018-01-01', '2020-12-31')
print(data)

pdf = pca(data)

dc, nc = optics(pdf)
cluster_pairs(0, dc, data)

