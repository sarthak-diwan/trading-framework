from backtest import BackTest, Strategy
import backtrader.indicators as btind
from common_utils import *
import statsmodels.api as sm
import backtrader as bt

def compute_hedge_ratio(data1, data2):
    data1_np = np.array(data1)
    data2_np = np.array(data2)
    model = sm.OLS(data1_np, sm.add_constant(data2_np)).fit()
    return float(model.params[1])  # Convert the hedge ratio to a scalar


class APOSpreadStrategy(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 40),
        ('buy_threshold', -10),
        ('sell_threshold', 10),
        ('paisa', 10000),
        ('max_pos', 100),
    )
    
    def __init__(self):
        # Calculate hedging ratio
        self.hedge_ratio = compute_hedge_ratio(list(self.datas[0].close), list(self.datas[1].close))
 
        # Create the spread series
        self.spread = self.datas[0].close - self.hedge_ratio * self.datas[1].close
        
        self.fast_ema = bt.indicators.EMA(self.spread, period=self.params.fast)
        self.slow_ema = bt.indicators.EMA(self.spread, period=self.params.slow)
        self.apo_spread = self.fast_ema - self.slow_ema
        self.our_pos = 0

        
    def next(self):
        stock1 = self.datas[0].close[0]
        stock2 = self.datas[1].close[0]
        x = self.p.paisa/(stock1 + self.hedge_ratio*stock2)
        if self.apo_spread[0] < self.params.buy_threshold and self.our_pos < self.p.max_pos:
            self.our_pos += 1
            self.buy(data=self.datas[0], size=x)   # Buy 1 share of stock 1
            self.sell(data=self.datas[1], size=x*self.hedge_ratio)  # Short hedge_ratio shares of stock 2

        elif self.apo_spread[0] > self.params.sell_threshold and self.our_pos > -self.p.max_pos:
            self.our_pos -= 1
            self.sell(data=self.datas[0], size=x)  # Short 1 share of stock 1
            self.buy(data=self.datas[1], size=x*self.hedge_ratio)   # Buy hedge_ratio shares of stock 2


if __name__ == "__main__":
    data = pd.read_csv('./data/cleaned_stocks.csv')
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)

    data_train, data_test = train_test_split(data, train_size=0.8)
    data=data_train
    pdf = pca(data, 3)
    f = open('results_train.txt', 'w')
    dc, nc = optics(pdf)
    for cluster in nc[1:]:
        try:
            cp = cluster_pairs(cluster, dc, data)
            if cp is None:
                continue
            print(cp['pair1'][0])
            # cp['pairs']
            n = len(cp)
            for i in range(n):
                # if cp['Cross'][i] < 25:
                #     continue
                stock1 = cp['pair1'][i].split('_')[0]
                stock2 = cp['pair2'][i].split('_')[0]
                # if(stock1 != "GESHIP" or stock2 != "SOMANYCERA"):
                #     continue
                xt = BackTest()
                xt.set_cash(100000)
                xt.add_symbols([stock1, stock2], test_train_split=True, train=True)
                xt.add_strategy(APOSpreadStrategy)
                results = xt.run()
                for result in results:
                    f.write(f"{stock1},{stock2},{result.analyzers.sharpe.get_analysis()['sharperatio']},{result.analyzers.drawdown.get_analysis()['drawdown']},{result.analyzers.returns.get_analysis()['rnorm100']},{result.hedge_ratio},{cp['Cointegration'][i]},{cp['Mean revertion'][i]},{cp['Half-life'][i]},{cp['Cross'][i]}\n")
                # xt.plot()
                print(f"Final Portfolio Value: {xt.get_value()}")
            print(cp)
        except Exception as e:
            print(e)
    print(dc)
