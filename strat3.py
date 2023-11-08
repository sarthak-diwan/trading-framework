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
        ('min_price_move', 10),
        ('min_profit_to_close', 10),
    )
    
    def __init__(self):
        # Calculate hedging ratio
        self.hedge_ratio = compute_hedge_ratio(list(self.datas[0].close), list(self.datas[1].close))
        
        # Create the spread series
        self.spread = self.datas[0].close - self.hedge_ratio * self.datas[1].close
        
        self.fast_ema = bt.indicators.EMA(self.spread, period=self.params.fast)
        self.slow_ema = bt.indicators.EMA(self.spread, period=self.params.slow)
        self.apo_spread = self.fast_ema - self.slow_ema
        
        self.last_buy_price = 0
        self.last_sell_price = 0

    def next(self):
        # Calculate the PnL for the current position
        current_pnl = self.broker.getvalue() - self.broker.get_cash()

        if ((self.apo_spread[0] > self.params.sell_threshold and abs(self.datas[0].close[0] - self.last_sell_price) > self.params.min_price_move) 
            or (self.position.size > 0 and (self.apo_spread[0] >= 0 or current_pnl > self.params.min_profit_to_close))):
            
            self.close(data=self.datas[0])  # Close position in stock 1
            self.close(data=self.datas[1])  # Close position in stock 2
            self.last_sell_price = self.datas[0].close[0]

        elif ((self.apo_spread[0] < self.params.buy_threshold and abs(self.datas[0].close[0] - self.last_buy_price) > self.params.min_price_move) 
            or (self.position.size < 0 and (self.apo_spread[0] <= 0 or current_pnl > self.params.min_profit_to_close))):
            
            self.close(data=self.datas[0])  # Close position in stock 1
            self.close(data=self.datas[1])  # Close position in stock 2
            self.last_buy_price = self.datas[0].close[0]
    
        # Entry logic remains the same as before
        elif not self.position:
            if self.apo_spread[0] < self.params.buy_threshold:
                self.buy(data=self.datas[0], size=10)   # Buy 1 share of stock 1
                self.sell(data=self.datas[1], size=10*self.hedge_ratio)  # Short hedge_ratio shares of stock 2
            elif self.apo_spread[0] > self.params.sell_threshold:
                self.sell(data=self.datas[0], size=10)  # Short 1 share of stock 1
                self.buy(data=self.datas[1], size=10*self.hedge_ratio)   # Buy hedge_ratio shares of stock 2

if __name__ == "__main__":
    data = pd.read_csv('./data/cleaned_stocks.csv')
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)

    data_train, data_test = train_test_split(data)
    data=data_train
    pdf = pca(data, 3)

    dc, nc = optics(pdf)
    for cluster in nc[1:]:
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
            xt = BackTest()
            xt.set_cash(100000)
            xt.add_symbols([stock1, stock2])
            xt.add_strategy(APOSpreadStrategy)
            xt.run()
            # xt.plot()
            print(f"Final Portfolio Value: {xt.get_value()}")
        print(cp)
    print(dc)