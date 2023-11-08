from backtest import BackTest, Strategy
import backtrader.indicators as btind
from common_utils import *
import statsmodels.api as sm
import backtrader as bt



class APOSpreadStrategy(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 40),
        ('buy_threshold', -10),
        ('sell_threshold', 10),
        ('paisa', 1000),
        ('max_pos', 100),
        ('hedge_ratio', 0.5)
    )
    
    def __init__(self):
        # Calculate hedging ratio
        self.hedge_ratio = self.p.hedge_ratio
        # Create the spread series
        self.spread = self.datas[0].close - self.hedge_ratio * self.datas[1].close
        # print(self.spread)
        self.fast_ema = bt.indicators.EMA(self.spread, period=self.params.fast)
        self.slow_ema = bt.indicators.EMA(self.spread, period=self.params.slow)
        self.apo_spread = self.fast_ema - self.slow_ema
        self.our_pos = 0

        
    def next(self):
        # print("Hedge Ratio: ", self.p.hedge_ratio)
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
    with open('results_train.txt', 'r') as f:
        data = f.readlines()
    
    data = [x.strip() for x in data]
    data = [x.split(',') for x in data]
    data = pd.DataFrame(data, columns=['pair1', 'pair2', 'train_sharpe', 'train_drawdown', 'train_returns', 'hedgeratio', 'Cointegration', 'Mean Reversion', 'Half-Life', 'Cross'])

    # Drop rows with share None
    data = data[data['train_sharpe'] != 'None']

    # Convert hedge ratio to float
    data['hedgeratio'] = data['hedgeratio'].astype(float)

    # only take values with sharpe > 0.5
    data = data[data['train_sharpe'].astype(float) > 0.5]

    # Sort the dataframe on sharpe
    data.sort_values(by=['train_sharpe'], inplace=True, ascending=False)

    # Sort the dataframe by drawdown
    data.sort_values(by=['train_drawdown'], inplace=True, ascending=True)

    data.reset_index(inplace=True, drop=True)
    f = open('results_test.txt', 'w')
    train_results = []
    for i in range(0, len(data)):
        stock1 = data['pair1'][i]
        stock2 = data['pair2'][i]
        xt = BackTest()
        xt.set_cash(100000)
        xt.add_symbols([stock1, stock2], test_train_split=True, train=False)
        xt.add_strategy(APOSpreadStrategy, hedge_ratio=float(data['hedgeratio'][i]))
        results = xt.run()

        for result in results:
            sharpe = result.analyzers.sharpe.get_analysis()['sharperatio']
            drawdown = result.analyzers.drawdown.get_analysis()['drawdown']
            returns = result.analyzers.returns.get_analysis()['rnorm100']

            f.write(f"{stock1},{stock2},{sharpe},{drawdown},{returns}\n")
            train_results.append([stock1, stock2, sharpe, drawdown, returns])

        # xt.plot()
        print(f"Final Portfolio Value: {xt.get_value()}")
    
    # make train_results a dataframe
    train_results = pd.DataFrame(train_results, columns=['pair1', 'pair2', 'test_sharpe', 'test_drawdown', 'test_returns'])

    # merge train results and data
    data = data.merge(train_results, on=['pair1', 'pair2'])

    data.to_csv('final_results_aman.csv', index=False)

    
    print(data.corr())